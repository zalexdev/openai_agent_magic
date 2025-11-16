"""
Example demonstrating transparent Tavily search integration.
The LLM automatically decides when to search for current information.
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def ask_current_question():
    """Ask a question that requires current information."""
    print("="*80)
    print("Question requiring current information")
    print("="*80 + "\n")

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    print("Question: What happened in the news today?")
    print("\nThe LLM will automatically use Tavily search to get current information.\n")

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "What are the top 3 news stories today?"}
        ]
    )

    print("Response:")
    print(response.choices[0].message.content)
    print("\n")


def ask_knowledge_question():
    """Ask a question the LLM can answer from its training."""
    print("="*80)
    print("Question from LLM's knowledge")
    print("="*80 + "\n")

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    print("Question: What is the capital of France?")
    print("\nThe LLM will answer directly without searching.\n")

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "What is the capital of France?"}
        ]
    )

    print("Response:")
    print(response.choices[0].message.content)
    print("\n")


def ask_specific_search_question():
    """Ask a question that definitely requires search."""
    print("="*80)
    print("Question requiring specific current data")
    print("="*80 + "\n")

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    print("Question: What is the current stock price of NVIDIA?")
    print("\nUsing Claude - it will search for current stock price.\n")

    response = client.chat.completions.create(
        model="claude-3-5-sonnet-20241022",
        messages=[
            {"role": "user", "content": "What is the current stock price of NVIDIA and what are analysts saying about it?"}
        ]
    )

    print("Response:")
    print(response.choices[0].message.content)
    print("\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Transparent Tavily Search Integration")
    print("="*80)
    print("\nYour LLMs now have access to web search - automatically!")
    print("The model decides when to use search based on the question.")
    print("No code changes needed - it just works!")
    print("="*80 + "\n")

    try:
        ask_current_question()
    except Exception as e:
        print(f"Current question test failed: {e}\n")

    try:
        ask_knowledge_question()
    except Exception as e:
        print(f"Knowledge question test failed: {e}\n")

    try:
        ask_specific_search_question()
    except Exception as e:
        print(f"Specific search test failed: {e}\n")
