"""
Streaming test example for the OpenAI-compatible API.
"""
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = "http://localhost:8000/v1/chat/completions"


def test_streaming():
    """Test streaming responses."""
    print("Testing streaming with OpenAI GPT-4...")
    print("Question: What are the top 3 tech news stories today?\n")

    response = requests.post(
        API_URL,
        json={
            "model": "gpt-4",
            "provider": "openai",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "tavily_api_key": os.getenv("TAVILY_API_KEY"),
            "messages": [
                {"role": "user", "content": "What are the top 3 tech news stories today?"}
            ],
            "stream": True,
            "temperature": 0.7
        },
        stream=True
    )

    print("Response:\n")
    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith('data: '):
                data_str = line_str[6:]  # Remove 'data: ' prefix

                if data_str == '[DONE]':
                    print("\n\nStream completed!")
                    break

                try:
                    data = json.loads(data_str)

                    # Extract content from the delta
                    if 'choices' in data and len(data['choices']) > 0:
                        delta = data['choices'][0].get('delta', {})
                        content = delta.get('content', '')

                        if content:
                            print(content, end='', flush=True)

                except json.JSONDecodeError:
                    pass


def test_streaming_anthropic():
    """Test streaming with Anthropic."""
    print("\n\n" + "="*80)
    print("Testing streaming with Anthropic Claude...")
    print("Question: Summarize the latest AI research breakthroughs\n")

    response = requests.post(
        API_URL,
        json={
            "model": "claude-3-5-sonnet-20241022",
            "provider": "anthropic",
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "tavily_api_key": os.getenv("TAVILY_API_KEY"),
            "messages": [
                {"role": "user", "content": "Summarize the latest AI research breakthroughs"}
            ],
            "stream": True,
            "max_tokens": 1024
        },
        stream=True
    )

    print("Response:\n")
    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith('data: '):
                data_str = line_str[6:]

                if data_str == '[DONE]':
                    print("\n\nStream completed!")
                    break

                try:
                    data = json.loads(data_str)

                    if 'choices' in data and len(data['choices']) > 0:
                        delta = data['choices'][0].get('delta', {})
                        content = delta.get('content', '')

                        if content:
                            print(content, end='', flush=True)

                except json.JSONDecodeError:
                    pass


if __name__ == "__main__":
    print("\n" + "="*80)
    print("OpenAI-Compatible API - Streaming Tests")
    print("="*80 + "\n")

    try:
        test_streaming()
    except Exception as e:
        print(f"\nStreaming test failed: {e}")

    try:
        test_streaming_anthropic()
    except Exception as e:
        print(f"\nAnthropic streaming test failed: {e}")
