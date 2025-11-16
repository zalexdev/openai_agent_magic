# OpenAI-Compatible API with Tavily Search

A powerful FastAPI-based service that provides an OpenAI-compatible API endpoint with **transparent Tavily search integration**. Use the standard OpenAI Python SDK with any LLM provider (OpenAI, Anthropic, Google) - just change the `base_url` and the model automatically detects the provider!

## 🎯 Key Features

- **100% OpenAI SDK Compatible**: Use the official OpenAI Python SDK - just change the base URL!
- **Auto-Provider Detection**: Automatically detects provider from model name (gpt-* → OpenAI, claude* → Anthropic, gemini* → Google)
- **Transparent Search Integration**: Tavily search automatically available to all models
- **Streaming Support**: Full support for streaming responses
- **Vision Support**: Multimodal content support (text + images)
- **Zero Code Changes**: Drop-in replacement - no changes to your existing OpenAI code!

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/zalexdev/openai_agent_magic.git
cd openai_agent_magic
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

4. Run the server:
```bash
python -m src.main
```

The API will be available at `http://localhost:8000`

## 📖 Usage

### Using with Official OpenAI SDK (Recommended)

The easiest way to use this API is with the official OpenAI Python SDK:

```python
from openai import OpenAI
import os

# Just change the base_url - everything else is standard OpenAI!
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key=os.getenv("OPENAI_API_KEY")  # Your provider's API key
)

# Use GPT-4 (auto-detected as OpenAI)
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "What's the latest news about AI?"}
    ]
)

print(response.choices[0].message.content)
```

### Using with Anthropic (Claude)

```python
from openai import OpenAI
import os

# Same client, different API key and model!
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# Use Claude (auto-detected as Anthropic)
response = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",
    messages=[
        {"role": "user", "content": "What are the latest developments in quantum computing?"}
    ]
)

print(response.choices[0].message.content)
```

### Using with Google (Gemini)

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Use Gemini (auto-detected as Google)
response = client.chat.completions.create(
    model="gemini-2.0-flash-exp",
    messages=[
        {"role": "user", "content": "What's the current price of Bitcoin?"}
    ]
)

print(response.choices[0].message.content)
```

### Streaming Responses

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Streaming works exactly like OpenAI!
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Explain quantum entanglement"}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end='', flush=True)
```

### Vision/Multimodal Support

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Vision works exactly like OpenAI!
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg"
                    }
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# API Keys (at least one provider + Tavily required)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
TAVILY_API_KEY=tvly-...

# Custom Base URL (optional - for using custom endpoints)
# If set, this overrides the default API endpoints for all providers
CUSTOM_BASE_URL=http://185.150.190.236:3000/v1

# Server Config
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info
```

### How It Works

1. **Set your base_url** to our API endpoint
2. **Use your provider's API key** in the client
3. **Choose any model** - we auto-detect the provider:
   - `gpt-*` → OpenAI
   - `claude*` → Anthropic
   - `gemini*` → Google
4. **Everything else is standard** OpenAI SDK!

### Request Parameters

Standard OpenAI parameters are supported:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model name (auto-detects provider) |
| `messages` | array | Yes | Array of message objects |
| `stream` | boolean | No | Enable streaming (default: false) |
| `temperature` | float | No | Sampling temperature 0-2 (default: 0.7) |
| `max_tokens` | integer | No | Maximum tokens to generate |
| `top_p` | float | No | Nucleus sampling parameter |
| `presence_penalty` | float | No | Presence penalty (-2.0 to 2.0) |
| `frequency_penalty` | float | No | Frequency penalty (-2.0 to 2.0) |
| `n` | integer | No | Number of completions to generate |
| `user` | string | No | Unique user identifier |

**Note**: API key is passed via the `Authorization` header (handled automatically by OpenAI SDK). Tavily API key is configured server-side in the `.env` file.

## 🏗️ Architecture

```
┌─────────────────┐
│   FastAPI App   │
│   (OpenAI API)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SearchAgent    │
│ (Tool Calling)  │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌─────────┐ ┌──────────┐
│   LLM   │ │  Tavily  │
│Provider │ │  Search  │
└─────────┘ └──────────┘
    │
    ├─ OpenAI
    ├─ Anthropic
    └─ Google
```

### Components

1. **FastAPI Application** (`main.py`): OpenAI-compatible REST API
2. **SearchAgent** (`agent.py`): LangChain agent with tool calling
3. **LLMProvider** (`llm_provider.py`): Multi-provider LLM wrapper
4. **Models** (`models.py`): Pydantic schemas for API validation

## 🔍 How It Works

1. **Request arrives** at the OpenAI-compatible endpoint
2. **LLMProvider** initializes the specified model (OpenAI/Anthropic/Google)
3. **Tavily search tool** is automatically bound to the model
4. **SearchAgent** processes the conversation:
   - Model decides if it needs to search
   - If needed, calls Tavily search automatically
   - Incorporates search results into response
5. **Response** is streamed or returned in OpenAI format

**The magic**: Users don't need to explicitly request search - the LLM decides when to use it!

## 📊 API Endpoints

### `POST /v1/chat/completions`

OpenAI-compatible chat completions endpoint.

**Request**:
```json
{
  "model": "gpt-4",
  "provider": "openai",
  "api_key": "sk-...",
  "tavily_api_key": "tvly-...",
  "messages": [
    {"role": "user", "content": "What's the weather like?"}
  ],
  "stream": false
}
```

**Response**:
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "gpt-4",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Based on current search results..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

### `GET /health`

Health check endpoint.

### `GET /`

API information and available endpoints.

## 🧪 Testing

Run the example scripts to test the API:

```bash
# Basic examples with OpenAI SDK
python examples/openai_sdk_basic.py

# Multi-provider examples (OpenAI, Anthropic, Google)
python examples/openai_sdk_multimodel.py

# Search integration examples
python examples/openai_sdk_search.py
```

Each example demonstrates different aspects:
- **openai_sdk_basic.py**: Basic usage, streaming, and vision
- **openai_sdk_multimodel.py**: Using all three providers with auto-detection
- **openai_sdk_search.py**: How the LLM automatically uses search when needed

## 🛠️ Development

### Project Structure

```
openai_agent_magic/
├── src/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   ├── models.py         # Pydantic models + auto-detection
│   ├── llm_provider.py   # Multi-provider LLM wrapper
│   └── agent.py          # Search agent with tool calling
├── examples/
│   ├── openai_sdk_basic.py      # Basic OpenAI SDK examples
│   ├── openai_sdk_multimodel.py # Multi-provider examples
│   ├── openai_sdk_search.py     # Search integration examples
│   ├── test_basic.py            # Direct HTTP examples
│   ├── test_streaming.py        # Streaming examples
│   └── test_vision.py           # Vision examples
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
├── .gitignore
├── run_server.sh        # Server startup script
└── README.md
```

### Running in Development

```bash
# With auto-reload
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Or using the main module
python -m src.main
```

## 🌟 Use Cases

1. **Research Assistant**: Ask about current events and get accurate, sourced information
2. **Market Analysis**: Get real-time data on stocks, crypto, markets
3. **Technical Support**: Search for latest documentation and solutions
4. **News Aggregation**: Summarize latest news on any topic
5. **Fact Checking**: Verify claims with current web data

## 🔒 Security Notes

- Never commit API keys to version control
- Use environment variables or secret management
- Implement rate limiting in production
- Add authentication/authorization as needed
- Validate and sanitize all inputs

## 📝 License

MIT License - feel free to use in your projects!

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📮 Support

For issues and questions, please use the GitHub issue tracker.

---

**Built with** ❤️ using FastAPI, LangChain, and Tavily Search
