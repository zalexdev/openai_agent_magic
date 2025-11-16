"""
Vision/multimodal test example for the OpenAI-compatible API.
"""
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = "http://localhost:8000/v1/chat/completions"


def test_vision_with_url():
    """Test vision capabilities with an image URL."""
    print("Testing vision with OpenAI GPT-4 Vision...")

    # Example with a public image URL
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

    response = requests.post(
        API_URL,
        json={
            "model": "gpt-4o",
            "provider": "openai",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "tavily_api_key": os.getenv("TAVILY_API_KEY"),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What's in this image? Describe it in detail."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 500
        }
    )

    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


def test_vision_with_question():
    """Test vision with a follow-up search question."""
    print("\n\n" + "="*80)
    print("Testing vision + search capability...")

    # Famous landmark image
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Paris_-_Eiffelturm_und_Marsfeld2.jpg/1280px-Paris_-_Eiffelturm_und_Marsfeld2.jpg"

    response = requests.post(
        API_URL,
        json={
            "model": "gpt-4o",
            "provider": "openai",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "tavily_api_key": os.getenv("TAVILY_API_KEY"),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What landmark is this? And when was it built? Search for current visitor information."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 800
        }
    )

    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    print("\n" + "="*80)
    print("OpenAI-Compatible API - Vision Tests")
    print("="*80 + "\n")

    try:
        test_vision_with_url()
    except Exception as e:
        print(f"Vision test failed: {e}")

    try:
        test_vision_with_question()
    except Exception as e:
        print(f"Vision + search test failed: {e}")
