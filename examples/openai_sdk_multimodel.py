"""
Example demonstrating auto-detection of different providers.
The same OpenAI SDK client can be used with any provider!
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def test_openai_model():
    """Test OpenAI model (gpt-*)."""
    print("="*80)
    print("Testing OpenAI (GPT-4)")
    print("="*80 + "\n")

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    response = client.chat.completions.create(
        model="gpt-4",  # Auto-detected as OpenAI
        messages=[
            {"role": "user", "content": "What are the top AI trends in 2025?"}
        ]
    )

    print(f"Model: {response.model}")
    print(f"Provider: Auto-detected as OpenAI (model starts with 'gpt-')")
    print(f"\nResponse:\n{response.choices[0].message.content}")
    print("\n")


def test_anthropic_model():
    """Test Anthropic model (claude*)."""
    print("="*80)
    print("Testing Anthropic (Claude)")
    print("="*80 + "\n")

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    response = client.chat.completions.create(
        model="claude-3-5-sonnet-20241022",  # Auto-detected as Anthropic
        messages=[
            {"role": "user", "content": "What are the latest developments in quantum computing?"}
        ]
    )

    print(f"Model: {response.model}")
    print(f"Provider: Auto-detected as Anthropic (model starts with 'claude')")
    print(f"\nResponse:\n{response.choices[0].message.content}")
    print("\n")


def test_google_model():
    """Test Google model (gemini*)."""
    print("="*80)
    print("Testing Google (Gemini)")
    print("="*80 + "\n")

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key=os.getenv("GOOGLE_API_KEY")
    )

    response = client.chat.completions.create(
        model="gemini-2.0-flash-exp",  # Auto-detected as Google
        messages=[
            {"role": "user", "content": "What's the current price of Bitcoin?"}
        ]
    )

    print(f"Model: {response.model}")
    print(f"Provider: Auto-detected as Google (model starts with 'gemini')")
    print(f"\nResponse:\n{response.choices[0].message.content}")
    print("\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Multi-Provider Example - Auto-Detection in Action")
    print("="*80)
    print("\nThe API automatically detects the provider from the model name:")
    print("  • gpt-* → OpenAI")
    print("  • claude* → Anthropic")
    print("  • gemini* → Google")
    print("\nJust change the model name and API key - everything else works the same!")
    print("="*80 + "\n")

    try:
        test_openai_model()
    except Exception as e:
        print(f"OpenAI test failed: {e}\n")

    try:
        test_anthropic_model()
    except Exception as e:
        print(f"Anthropic test failed: {e}\n")

    try:
        test_google_model()
    except Exception as e:
        print(f"Google test failed: {e}\n")
