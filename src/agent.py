"""
Agent wrapper for LLM with Tavily search integration.
Uses the new LangChain create_agent API.
"""
from typing import List, AsyncIterator, Optional, Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from .llm_provider import LLMProvider
from .models import Message


class SearchAgent:
    """
    Agent that transparently integrates Tavily search with any LLM.
    Users don't need to explicitly request search - the agent decides when to use it.
    """

    DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant with access to web search capabilities via Tavily.

When answering questions:
- Use the tavily_search tool to find current information when needed
- Cite sources when using search results
- If you don't need to search, answer directly from your knowledge
- Be concise and accurate in your responses"""

    def __init__(
        self,
        llm_provider: LLMProvider,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the search agent.

        Args:
            llm_provider: LLM provider instance with tools configured
            system_prompt: Optional custom system prompt
        """
        self.llm_provider = llm_provider
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.agent: Optional[Runnable] = None

        # Create agent if tools are available
        if self.llm_provider.tools:
            self._create_agent()

    def _create_agent(self):
        """Create the agent with Tavily search using new LangChain API."""
        try:
            from langchain.agents import create_agent

            self.agent = create_agent(
                model=self.llm_provider.llm,
                tools=self.llm_provider.tools,
                system_prompt=self.system_prompt
            )
        except ImportError:
            # Fallback: just use LLM with bound tools
            self.agent = None

    async def generate(self, messages: List[Message]) -> AIMessage:
        """
        Generate a response, using search when needed.
        Falls back to direct LLM if no agent is available.
        """
        langchain_messages = self.llm_provider.convert_messages(messages)

        if self.agent:
            # Use the agent with tool calling
            result = await self.agent.ainvoke({
                "messages": langchain_messages
            })

            # Extract the final message from the result
            if isinstance(result, dict) and "messages" in result:
                final_message = result["messages"][-1]
                return final_message
            else:
                # Fallback if result format is different
                return AIMessage(content=str(result))
        else:
            # Direct LLM call with tools bound
            return await self.llm_provider.generate(messages)

    async def stream(self, messages: List[Message]) -> AsyncIterator[str]:
        """
        Stream responses, using search when needed.
        """
        langchain_messages = self.llm_provider.convert_messages(messages)

        if self.agent:
            # Stream from the agent
            try:
                async for chunk in self.agent.astream({
                    "messages": langchain_messages
                }, stream_mode="values"):
                    # Extract the latest message from the chunk
                    if isinstance(chunk, dict) and "messages" in chunk:
                        latest_message = chunk["messages"][-1]
                        if hasattr(latest_message, "content") and latest_message.content:
                            # Only yield if content is new
                            if isinstance(latest_message.content, str):
                                yield latest_message.content
            except Exception:
                # Fallback to direct streaming if agent streaming fails
                async for chunk in self.llm_provider.stream(messages):
                    if hasattr(chunk, "content") and chunk.content:
                        yield chunk.content
        else:
            # Direct LLM streaming
            async for chunk in self.llm_provider.stream(messages):
                if hasattr(chunk, "content") and chunk.content:
                    yield chunk.content

    async def stream_with_tool_info(
        self,
        messages: List[Message]
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream responses with tool call information.
        Provides visibility into when search is being used.
        """
        langchain_messages = self.llm_provider.convert_messages(messages)

        if not self.agent:
            # No tools, just stream content
            async for chunk in self.llm_provider.stream(messages):
                if hasattr(chunk, "content") and chunk.content:
                    yield {
                        "type": "content",
                        "content": chunk.content
                    }
            return

        try:
            async for event in self.agent.astream_events(
                {"messages": langchain_messages},
                version="v2"
            ):
                kind = event.get("event")

                # Tool start
                if kind == "on_tool_start":
                    tool_name = event.get("name")
                    tool_input = event.get("data", {}).get("input")
                    yield {
                        "type": "tool_start",
                        "tool": tool_name,
                        "input": tool_input
                    }

                # Tool end
                elif kind == "on_tool_end":
                    tool_name = event.get("name")
                    tool_output = event.get("data", {}).get("output")
                    yield {
                        "type": "tool_end",
                        "tool": tool_name,
                        "output": tool_output
                    }

                # LLM content streaming
                elif kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        yield {
                            "type": "content",
                            "content": chunk.content
                        }
        except Exception:
            # Fallback to simple streaming
            async for chunk in self.llm_provider.stream(messages):
                if hasattr(chunk, "content") and chunk.content:
                    yield {
                        "type": "content",
                        "content": chunk.content
                    }
