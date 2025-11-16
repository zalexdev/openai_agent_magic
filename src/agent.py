"""
Agent wrapper for LLM with Tavily search integration.
Handles tool calling and agent execution.
"""
from typing import List, AsyncIterator, Optional, Dict, Any
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_tavily import TavilySearch

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
- Be concise and accurate in your responses

Available tools:
{tools}
"""

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

        # Create agent if tools are available
        if self.llm_provider.tools:
            self.agent_executor = self._create_agent()
        else:
            self.agent_executor = None

    def _create_agent(self) -> AgentExecutor:
        """Create the tool-calling agent with Tavily search."""

        # Create prompt template with system message and chat history
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create the tool calling agent
        agent = create_tool_calling_agent(
            llm=self.llm_provider.llm,
            tools=self.llm_provider.tools,
            prompt=prompt
        )

        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.llm_provider.tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=5,
        )

        return agent_executor

    def _prepare_agent_input(self, messages: List[Message]) -> Dict[str, Any]:
        """
        Prepare input for agent from messages.
        Separates the latest user message from chat history.
        """
        langchain_messages = self.llm_provider.convert_messages(messages)

        # Get the last message as input
        if langchain_messages:
            last_message = langchain_messages[-1]
            chat_history = langchain_messages[:-1] if len(langchain_messages) > 1 else []

            # Extract text content from last message
            if isinstance(last_message.content, str):
                input_text = last_message.content
            elif isinstance(last_message.content, list):
                # Handle multimodal content - extract text parts
                text_parts = [
                    part.get("text", "")
                    for part in last_message.content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                input_text = " ".join(text_parts)
            else:
                input_text = str(last_message.content)
        else:
            input_text = ""
            chat_history = []

        return {
            "input": input_text,
            "chat_history": chat_history,
        }

    async def generate(self, messages: List[Message]) -> AIMessage:
        """
        Generate a response, using search when needed.
        Falls back to direct LLM if no agent is available.
        """
        if self.agent_executor:
            # Use agent with tool calling
            agent_input = self._prepare_agent_input(messages)
            result = await self.agent_executor.ainvoke(agent_input)

            # Convert result to AIMessage
            output = result.get("output", "")
            return AIMessage(content=output)
        else:
            # Direct LLM call without tools
            return await self.llm_provider.generate(messages)

    async def stream(self, messages: List[Message]) -> AsyncIterator[str]:
        """
        Stream responses, using search when needed.

        Note: Agent streaming is more complex as it involves multiple tool calls.
        For simplicity, we'll use the direct LLM streaming with tools bound.
        """
        if self.agent_executor:
            # For agent with tools, we need to handle streaming differently
            # Since agent execution involves multiple steps, we'll invoke then stream
            agent_input = self._prepare_agent_input(messages)

            # Stream agent events
            async for event in self.agent_executor.astream_events(
                agent_input,
                version="v1"
            ):
                kind = event.get("event")

                # Stream LLM token chunks
                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content"):
                        if chunk.content:
                            yield chunk.content

                # Stream final output
                elif kind == "on_chain_end":
                    if event.get("name") == "AgentExecutor":
                        output = event.get("data", {}).get("output", {})
                        if isinstance(output, dict):
                            final_output = output.get("output", "")
                            if final_output:
                                yield final_output
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
        if not self.agent_executor:
            # No tools, just stream content
            async for chunk in self.llm_provider.stream(messages):
                if hasattr(chunk, "content") and chunk.content:
                    yield {
                        "type": "content",
                        "content": chunk.content
                    }
            return

        agent_input = self._prepare_agent_input(messages)

        async for event in self.agent_executor.astream_events(
            agent_input,
            version="v1"
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
