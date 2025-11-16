"""
OpenAI-compatible API models for chat completions with Tavily search integration.
"""
from typing import List, Optional, Dict, Any, Literal, Union
from pydantic import BaseModel, Field
from enum import Enum


class ModelProvider(str, Enum):
    """Supported model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class MessageRole(str, Enum):
    """Message roles in chat completion."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ImageUrl(BaseModel):
    """Image URL for vision models."""
    url: str = Field(..., description="URL of the image or base64 encoded image")
    detail: Optional[str] = Field(default="auto", description="Detail level: low, high, or auto")


class ContentPart(BaseModel):
    """Content part (text or image) for multimodal messages."""
    type: Literal["text", "image_url"]
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None


class Message(BaseModel):
    """Chat message compatible with OpenAI format."""
    role: MessageRole
    content: Union[str, List[ContentPart]] = Field(..., description="Message content (text or multimodal)")
    name: Optional[str] = Field(default=None, description="Optional name for the message")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(default=None, description="Tool calls made by assistant")
    tool_call_id: Optional[str] = Field(default=None, description="ID of the tool call this message is responding to")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request with provider configuration."""

    # Standard OpenAI parameters
    model: str = Field(..., description="Model name to use")
    messages: List[Message] = Field(..., description="List of messages in the conversation")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    stream: Optional[bool] = Field(default=False, description="Whether to stream responses")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="Stop sequences")
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)

    # Provider-specific configuration
    provider: ModelProvider = Field(..., description="Model provider: openai, anthropic, or google")
    api_key: str = Field(..., description="API key for the model provider")
    api_base: Optional[str] = Field(default=None, description="Optional custom API base URL")

    # Tavily search configuration
    tavily_api_key: Optional[str] = Field(default=None, description="Tavily API key for search functionality")
    enable_search: Optional[bool] = Field(default=True, description="Enable Tavily search tool")
    max_search_results: Optional[int] = Field(default=5, ge=1, le=10, description="Max search results per query")

    class Config:
        use_enum_values = True


class Usage(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    """A completion choice."""
    index: int
    message: Message
    finish_reason: Optional[str] = Field(default=None, description="Reason for completion finish")


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str = Field(..., description="Unique completion ID")
    object: str = Field(default="chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for completion")
    choices: List[Choice]
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None


class StreamChoice(BaseModel):
    """A streaming completion choice."""
    index: int
    delta: Dict[str, Any] = Field(..., description="Incremental message delta")
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    """OpenAI-compatible streaming response chunk."""
    id: str
    object: str = Field(default="chat.completion.chunk")
    created: int
    model: str
    choices: List[StreamChoice]
    system_fingerprint: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response model."""
    error: Dict[str, Any] = Field(..., description="Error details")

    class Config:
        schema_extra = {
            "example": {
                "error": {
                    "message": "Invalid API key provided",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key"
                }
            }
        }
