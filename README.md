# OpenAI-Compatible API with Tavily Search

A powerful FastAPI-based service that provides an OpenAI-compatible API endpoint with **transparent Tavily search integration**. This allows you to add web search capabilities to any LLM (OpenAI, Anthropic, Google) without changing your existing code!

## 🎯 Key Features

- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's chat completions endpoint
- **Multi-Provider Support**: Works with OpenAI, Anthropic (Claude), and Google (Gemini)
- **Transparent Search Integration**: Tavily search automatically available to all models
- **Streaming Support**: Full support for streaming responses
- **Vision Support**: Multimodal content support (text + images)
- **No Code Changes Required**: Users and developers don't need to modify their code - search works transparently!

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

### Basic Example (OpenAI)

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "gpt-4",
        "provider": "openai",
        "api_key": "your-openai-api-key",
        "tavily_api_key": "your-tavily-api-key",
        "messages": [
            {"role": "user", "content": "What's the latest news about AI?"}
        ],
        "stream": False
    }
)

print(response.json())
```

### Using with Anthropic (Claude)

```python
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "claude-3-5-sonnet-20241022",
        "provider": "anthropic",
        "api_key": "your-anthropic-api-key",
        "tavily_api_key": "your-tavily-api-key",
        "messages": [
            {"role": "user", "content": "What are the latest developments in quantum computing?"}
        ]
    }
)
```

### Using with Google (Gemini)

```python
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "gemini-2.0-flash-exp",
        "provider": "google",
        "api_key": "your-google-api-key",
        "tavily_api_key": "your-tavily-api-key",
        "messages": [
            {"role": "user", "content": "What's the current price of Bitcoin?"}
        ]
    }
)
```

### Streaming Responses

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "gpt-4",
        "provider": "openai",
        "api_key": "your-openai-api-key",
        "tavily_api_key": "your-tavily-api-key",
        "messages": [
            {"role": "user", "content": "Explain quantum entanglement"}
        ],
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

### Vision/Multimodal Support

```python
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "gpt-4o",
        "provider": "openai",
        "api_key": "your-openai-api-key",
        "tavily_api_key": "your-tavily-api-key",
        "messages": [
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
    }
)
```

### Using with OpenAI Python SDK

You can use this API as a drop-in replacement for OpenAI:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # Not used by our proxy
)

# Note: You'll need to pass provider and keys in messages or modify the client
# For full compatibility, use the requests library as shown above
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

# Server Config
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info
```

### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model name (e.g., "gpt-4", "claude-3-5-sonnet-20241022") |
| `provider` | string | Yes | Provider: "openai", "anthropic", or "google" |
| `api_key` | string | Yes | API key for the model provider |
| `messages` | array | Yes | Array of message objects |
| `tavily_api_key` | string | No* | Tavily API key (*required for search) |
| `stream` | boolean | No | Enable streaming (default: false) |
| `temperature` | float | No | Sampling temperature 0-2 (default: 0.7) |
| `max_tokens` | integer | No | Maximum tokens to generate |
| `enable_search` | boolean | No | Enable Tavily search (default: true) |
| `max_search_results` | integer | No | Max search results 1-10 (default: 5) |

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

Create a test script `test_api.py`:

```python
import requests
import json

def test_chat():
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "provider": "openai",
            "api_key": "your-key",
            "tavily_api_key": "your-tavily-key",
            "messages": [
                {"role": "user", "content": "What's the latest on SpaceX?"}
            ]
        }
    )
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test_chat()
```

## 🛠️ Development

### Project Structure

```
openai_agent_magic/
├── src/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   ├── models.py         # Pydantic models
│   ├── llm_provider.py   # Multi-provider LLM wrapper
│   └── agent.py          # Search agent with tool calling
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
├── .gitignore
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
