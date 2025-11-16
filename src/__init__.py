"""
OpenAI-compatible API with Tavily Search integration.
"""
from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Message,
    ModelProvider,
)
from .llm_provider import LLMProvider
from .agent import SearchAgent

__version__ = "1.0.0"

__all__ = [
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "Message",
    "ModelProvider",
    "LLMProvider",
    "SearchAgent",
]
