"""
Example using the official OpenAI Python SDK with our API.
This demonstrates how our API is a drop-in replacement for OpenAI's API.
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create OpenAI client pointing to our API
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key=os.getenv("OPENAI_API_KEY")  # Your OpenAI API key
)


def test_gpt_basic():
    """Test basic GPT request."""
    print("Testing GPT-4 with Tavily search...")
    print("Question: What's the latest news about SpaceX?\n")

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "What's the latest news about SpaceX?"}
        ],
        temperature=0.7
    )

    print("Response:")
    print(response.choices[0].message.content)
    print("\n" + "="*80 + "\n")


def test_gpt_streaming():
    """Test streaming with GPT."""
    print("Testing GPT-4 streaming...")
    print("Question: Explain quantum computing\n")

    print("Response (streaming):\n")

    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "Explain quantum computing in simple terms"}
        ],
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end='', flush=True)

    print("\n\n" + "="*80 + "\n")


def test_gpt_vision():
    """Test vision capabilities with GPT-4 Vision."""
    print("Testing GPT-4 Vision...")

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
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                        }
                    }
                ]
            }
        ],
        max_tokens=300
    )

    print("Response:")
    print(response.choices[0].message.content)
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("OpenAI SDK Examples - Using our API as a drop-in replacement")
    print("="*80 + "\n")

    try:
        test_gpt_basic()
    except Exception as e:
        print(f"Basic test failed: {e}\n")

    try:
        test_gpt_streaming()
    except Exception as e:
        print(f"Streaming test failed: {e}\n")

    try:
        test_gpt_vision()
    except Exception as e:
        print(f"Vision test failed: {e}\n")
