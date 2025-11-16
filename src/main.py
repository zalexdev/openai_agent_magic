"""
FastAPI application providing OpenAI-compatible API with Tavily search integration.
"""
import time
import json
import uuid
from typing import AsyncIterator
from fastapi import FastAPI, HTTPException, Request
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
            "health": "/health"
        },
        "supported_providers": ["openai", "anthropic", "google"],
        "features": ["streaming", "vision", "tool_calling", "tavily_search"]
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": int(time.time())}


def create_error_response(message: str, error_type: str = "invalid_request_error") -> JSONResponse:
    """Create standardized error response."""
    return JSONResponse(
        status_code=400,
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


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    Transparently adds Tavily search to any model.
    """
    try:
        # Validate request
        if not request.messages:
            return create_error_response("messages field is required and cannot be empty")

        # Create LLM provider
        try:
            llm_provider = LLMProvider(
                provider=request.provider,
                model_name=request.model,
                api_key=request.api_key,
                api_base=request.api_base,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                tavily_api_key=request.tavily_api_key,
                enable_search=request.enable_search,
                max_search_results=request.max_search_results,
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
                    messages=request.messages,
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
            response_message = await agent.generate(request.messages)

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
            prompt_tokens = sum(len(str(m.content).split()) for m in request.messages) * 2
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
