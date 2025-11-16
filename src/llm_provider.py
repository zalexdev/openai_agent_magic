"""
Multi-provider LLM wrapper with Tavily search integration.
Supports OpenAI, Anthropic, and Google models with unified interface.
"""
import os
from typing import Optional, List, Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool

from .models import ModelProvider, Message, MessageRole, ContentPart


class LLMProvider:
    """
    Unified LLM provider interface supporting multiple backends.
    Automatically integrates Tavily search tool for web search capabilities.
    """

    def __init__(
        self,
        provider: ModelProvider,
        model_name: str,
        api_key: str,
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tavily_api_key: Optional[str] = None,
        enable_search: bool = True,
        max_search_results: int = 5,
        **kwargs
    ):
        """
        Initialize LLM provider with specified backend.

        Args:
            provider: Model provider (openai, anthropic, google)
            model_name: Name of the model to use
            api_key: API key for the model provider
            api_base: Optional custom API base URL
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tavily_api_key: Tavily API key for search functionality
            enable_search: Whether to enable Tavily search tool
            max_search_results: Maximum search results per query
            **kwargs: Additional provider-specific parameters
        """
        self.provider = provider
        self.model_name = model_name
        self.enable_search = enable_search
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")

        # Initialize the appropriate chat model
        self.llm = self._create_llm(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        # Initialize Tavily search tool if enabled
        self.tools: List[BaseTool] = []
        if enable_search and self.tavily_api_key:
            self.tools.append(
                TavilySearch(
                    api_key=self.tavily_api_key,
                    max_results=max_search_results,
                    include_answer=True,
                    include_raw_content=True,
                )
            )

        # Bind tools to the LLM if available
        if self.tools:
            self.llm_with_tools = self.llm.bind_tools(self.tools)
        else:
            self.llm_with_tools = self.llm

    def _create_llm(
        self,
        provider: ModelProvider,
        model_name: str,
        api_key: str,
        api_base: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> BaseChatModel:
        """Create the appropriate LLM instance based on provider."""

        if provider == ModelProvider.OPENAI:
            llm_kwargs = {
                "model": model_name,
                "api_key": api_key,
                "temperature": temperature,
            }
            if api_base:
                llm_kwargs["base_url"] = api_base
                print(f"[LLMProvider] Using custom base URL for OpenAI: {api_base}")
            if max_tokens:
                llm_kwargs["max_tokens"] = max_tokens

            return ChatOpenAI(**llm_kwargs, **kwargs)

        elif provider == ModelProvider.ANTHROPIC:
            llm_kwargs = {
                "model": model_name,
                "anthropic_api_key": api_key,
                "temperature": temperature,
            }
            if api_base:
                llm_kwargs["base_url"] = api_base
                print(f"[LLMProvider] Using custom base URL for Anthropic: {api_base}")
            if max_tokens:
                llm_kwargs["max_tokens"] = max_tokens

            return ChatAnthropic(**llm_kwargs, **kwargs)

        elif provider == ModelProvider.GOOGLE:
            llm_kwargs = {
                "model": model_name,
                "google_api_key": api_key,
                "temperature": temperature,
            }
            # Google Generative AI doesn't support custom base URLs via LangChain
            # For custom endpoints with Google models, use OpenAI-compatible mode
            if api_base:
                print(f"[LLMProvider] Custom base URL for Google: {api_base}")
                print("[LLMProvider] Note: Using OpenAI compatibility mode for custom endpoint")
                # Use ChatOpenAI with the custom base URL for Google models
                return ChatOpenAI(
                    model=model_name,
                    api_key=api_key,
                    base_url=api_base,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )

            if max_tokens:
                llm_kwargs["max_output_tokens"] = max_tokens

            return ChatGoogleGenerativeAI(**llm_kwargs, **kwargs)

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def convert_messages(self, messages: List[Message]) -> List[BaseMessage]:
        """
        Convert OpenAI-format messages to LangChain format.
        Supports multimodal content (text + images).
        """
        langchain_messages = []

        for msg in messages:
            # Handle multimodal content
            if isinstance(msg.content, list):
                # Convert list of content parts to appropriate format
                content_parts = []
                for part in msg.content:
                    if isinstance(part, ContentPart):
                        if part.type == "text" and part.text:
                            content_parts.append({"type": "text", "text": part.text})
                        elif part.type == "image_url" and part.image_url:
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {"url": part.image_url.url}
                            })
                    elif isinstance(part, dict):
                        content_parts.append(part)

                # Create message with multimodal content
                if msg.role == MessageRole.USER:
                    langchain_messages.append(HumanMessage(content=content_parts))
                elif msg.role == MessageRole.ASSISTANT:
                    langchain_messages.append(AIMessage(content=content_parts))
                elif msg.role == MessageRole.SYSTEM:
                    langchain_messages.append(SystemMessage(content=content_parts))
            else:
                # Handle simple text content
                content = msg.content if isinstance(msg.content, str) else str(msg.content)

                if msg.role == MessageRole.SYSTEM:
                    langchain_messages.append(SystemMessage(content=content))
                elif msg.role == MessageRole.USER:
                    langchain_messages.append(HumanMessage(content=content))
                elif msg.role == MessageRole.ASSISTANT:
                    langchain_messages.append(AIMessage(content=content))

        return langchain_messages

    async def generate(self, messages: List[Message]) -> AIMessage:
        """
        Generate a response for the given messages.
        Automatically uses Tavily search when needed.
        """
        langchain_messages = self.convert_messages(messages)
        response = await self.llm_with_tools.ainvoke(langchain_messages)
        return response

    async def stream(self, messages: List[Message]):
        """
        Stream responses for the given messages.
        Automatically uses Tavily search when needed.
        """
        langchain_messages = self.convert_messages(messages)
        async for chunk in self.llm_with_tools.astream(langchain_messages):
            yield chunk

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration."""
        return {
            "provider": self.provider.value,
            "model": self.model_name,
            "tools_enabled": len(self.tools) > 0,
            "available_tools": [tool.name for tool in self.tools]
        }
