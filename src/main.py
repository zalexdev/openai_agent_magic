"""
FastAPI application providing OpenAI-compatible API with Tavily search integration.
"""
import os
import time
import json
import uuid
from typing import AsyncIterator, Optional
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    Choice,
    StreamChoice,
    Message,
    MessageRole,
    Usage,
    ErrorResponse,
    detect_provider_from_model,
)
from .llm_provider import LLMProvider
from .agent import SearchAgent


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("🚀 OpenAI-compatible API with Tavily Search starting up...")
    yield
    # Shutdown
    print("👋 Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="OpenAI-Compatible API with Tavily Search",
    description="Transparent Tavily search integration for OpenAI, Anthropic, and Google models",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "OpenAI-Compatible API with Tavily Search",
        "version": "1.0.0",
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "responses": "/v1/responses",
            "responses_alt": "/responses",
            "health": "/health"
        },
        "supported_providers": ["openai", "anthropic", "google"],
        "features": ["streaming", "vision", "tool_calling", "tavily_search", "auto_provider_detection"],
        "note": "Both /v1/chat/completions and /v1/responses endpoints are supported"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": int(time.time())}


def extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    """
    Extract API key from Authorization header.

    Args:
        authorization: Authorization header value (e.g., "Bearer sk-...")

    Returns:
        API key or None
    """
    if not authorization:
        return None

    # Check if it's a Bearer token
    if authorization.startswith("Bearer "):
        return authorization[7:]  # Remove "Bearer " prefix

    # Also accept the token without "Bearer" prefix
    return authorization


def create_error_response(
    message: str,
    error_type: str = "invalid_request_error",
    status_code: int = 400
) -> JSONResponse:
    """Create standardized error response."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "code": None
            }
        }
    )


async def stream_completion(
    agent: SearchAgent,
    messages: list[Message],
    model: str,
    request_id: str,
) -> AsyncIterator[str]:
    """
    Stream chat completion in OpenAI format.
    """
    created = int(time.time())

    # Send initial chunk
    initial_chunk = ChatCompletionStreamResponse(
        id=request_id,
        created=created,
        model=model,
        choices=[
            StreamChoice(
                index=0,
                delta={"role": "assistant", "content": ""},
                finish_reason=None
            )
        ]
    )
    yield f"data: {initial_chunk.model_dump_json()}\n\n"

    # Stream content chunks
    try:
        async for content in agent.stream(messages):
            if content:
                chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    created=created,
                    model=model,
                    choices=[
                        StreamChoice(
                            index=0,
                            delta={"content": content},
                            finish_reason=None
                        )
                    ]
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

        # Send final chunk with finish_reason
        final_chunk = ChatCompletionStreamResponse(
            id=request_id,
            created=created,
            model=model,
            choices=[
                StreamChoice(
                    index=0,
                    delta={},
                    finish_reason="stop"
                )
            ]
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "internal_error"
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"


async def handle_completion_request(
    request: ChatCompletionRequest,
    authorization: Optional[str]
):
    """
    Handle completion request (shared logic for multiple endpoints).

    The API key should be provided in the Authorization header:
    Authorization: Bearer YOUR_API_KEY

    The provider is auto-detected from the model name:
    - gpt-* → OpenAI
    - claude* → Anthropic
    - gemini* → Google
    """
    try:
        # Get messages from either 'messages' or 'input' field
        messages = request.get_messages()

        # Validate request
        if not messages:
            return create_error_response("messages or input field is required and cannot be empty")

        # Extract API key from Authorization header
        api_key = extract_bearer_token(authorization)
        if not api_key:
            return create_error_response(
                "You didn't provide an API key. You need to provide your API key in an Authorization header using Bearer auth (i.e. Authorization: Bearer YOUR_KEY)",
                error_type="invalid_request_error",
                status_code=401
            )

        # Auto-detect provider from model name
        provider = detect_provider_from_model(request.model)
        print(f"[API] Detected provider: {provider.value} for model: {request.model}")

        # Get configuration from environment
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        custom_base_url = os.getenv("CUSTOM_BASE_URL")  # e.g., http://185.150.190.236:3000/v1

        # Debug logging
        if custom_base_url:
            print(f"[API] Using custom base URL: {custom_base_url}")
        else:
            print(f"[API] No custom base URL set, using default for {provider.value}")

        # Create LLM provider
        try:
            llm_provider = LLMProvider(
                provider=provider,
                model_name=request.model,
                api_key=api_key,
                api_base=custom_base_url,  # Use custom base URL if provided, else defaults
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                tavily_api_key=tavily_api_key,
                enable_search=bool(tavily_api_key),  # Enable search only if Tavily key is available
                max_search_results=5,
                top_p=request.top_p,
            )
        except Exception as e:
            return create_error_response(f"Failed to initialize LLM provider: {str(e)}")

        # Create agent with search capabilities
        agent = SearchAgent(llm_provider=llm_provider)

        # Generate request ID
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        # Handle streaming
        if request.stream:
            return StreamingResponse(
                stream_completion(
                    agent=agent,
                    messages=messages,
                    model=request.model,
                    request_id=request_id,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
            )

        # Non-streaming response
        try:
            response_message = await agent.generate(messages)

            # Extract content from response
            if hasattr(response_message, 'content'):
                content = response_message.content
            else:
                content = str(response_message)

            # Create response message
            assistant_message = Message(
                role=MessageRole.ASSISTANT,
                content=content
            )

            # Estimate token usage (simplified - real implementation would use tokenizers)
            prompt_tokens = sum(len(str(m.content).split()) for m in messages) * 2
            completion_tokens = len(str(content).split()) * 2
            total_tokens = prompt_tokens + completion_tokens

            response = ChatCompletionResponse(
                id=request_id,
                created=int(time.time()),
                model=request.model,
                choices=[
                    Choice(
                        index=0,
                        message=assistant_message,
                        finish_reason="stop"
                    )
                ],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens
                )
            )

            return response

        except Exception as e:
            return create_error_response(f"Error generating response: {str(e)}", "api_error")

    except Exception as e:
        return create_error_response(f"Unexpected error: {str(e)}", "internal_error")


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None)
):
    """
    OpenAI chat completions endpoint.
    Standard endpoint for chat-based models.
    """
    return await handle_completion_request(request, authorization)


@app.post("/v1/responses")
async def responses(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None)
):
    """
    OpenAI Responses API endpoint.
    Newer endpoint format that's compatible with the latest OpenAI SDK.
    """
    return await handle_completion_request(request, authorization)


@app.post("/responses")
async def responses_no_version(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Responses endpoint without /v1 prefix (for compatibility).
    """
    return await handle_completion_request(request, authorization)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": f"Internal server error: {str(exc)}",
                "type": "internal_error",
                "code": None
            }
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
