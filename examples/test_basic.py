"""
Basic test example for the OpenAI-compatible API with Tavily search.
"""
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_URL = "http://localhost:8000/v1/chat/completions"

def test_openai_basic():
    """Test basic OpenAI request."""
    print("Testing OpenAI (GPT-4)...")

    response = requests.post(
        API_URL,
        json={
            "model": "gpt-4",
            "provider": "openai",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "tavily_api_key": os.getenv("TAVILY_API_KEY"),
            "messages": [
                {"role": "user", "content": "What's the latest news about SpaceX launches?"}
            ],
            "temperature": 0.7,
            "stream": False
        }
    )

    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print("\n" + "="*80 + "\n")


def test_anthropic_basic():
    """Test basic Anthropic request."""
    print("Testing Anthropic (Claude)...")

    response = requests.post(
        API_URL,
        json={
            "model": "claude-3-5-sonnet-20241022",
            "provider": "anthropic",
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "tavily_api_key": os.getenv("TAVILY_API_KEY"),
            "messages": [
                {"role": "user", "content": "What are the latest developments in quantum computing?"}
            ],
            "max_tokens": 1024
        }
    )

    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print("\n" + "="*80 + "\n")


def test_google_basic():
    """Test basic Google request."""
    print("Testing Google (Gemini)...")

    response = requests.post(
        API_URL,
        json={
            "model": "gemini-2.0-flash-exp",
            "provider": "google",
            "api_key": os.getenv("GOOGLE_API_KEY"),
            "tavily_api_key": os.getenv("TAVILY_API_KEY"),
            "messages": [
                {"role": "user", "content": "What's the current price of Bitcoin?"}
            ]
        }
    )

    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print("\n" + "="*80 + "\n")


def test_without_search():
    """Test request without search enabled."""
    print("Testing without search...")

    response = requests.post(
        API_URL,
        json={
            "model": "gpt-3.5-turbo",
            "provider": "openai",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "enable_search": False,
            "messages": [
                {"role": "user", "content": "Explain quantum entanglement in simple terms"}
            ]
        }
    )

    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("OpenAI-Compatible API with Tavily Search - Basic Tests")
    print("="*80 + "\n")

    # Run tests
    try:
        test_openai_basic()
    except Exception as e:
        print(f"OpenAI test failed: {e}\n")

    try:
        test_anthropic_basic()
    except Exception as e:
        print(f"Anthropic test failed: {e}\n")

    try:
        test_google_basic()
    except Exception as e:
        print(f"Google test failed: {e}\n")

    try:
        test_without_search()
    except Exception as e:
        print(f"No-search test failed: {e}\n")
