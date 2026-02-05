"""Simple chat example.

Demonstrates basic chat functionality with AIX.

Usage:
    python examples/01_simple_chat.py
"""

from aix import chat


def main():
    """Run simple chat examples."""
    print("=" * 60)
    print("AIX Simple Chat Examples")
    print("=" * 60)

    # Example 1: Simple question
    print("\n1. Simple Question:")
    print("-" * 40)
    response = chat("What is 2+2?")
    print(f"Response: {response}")

    # Example 2: With specific model
    print("\n2. With Specific Model (gpt-4o-mini):")
    print("-" * 40)
    response = chat("Explain quantum computing in one sentence.", model="gpt-4o-mini")
    print(f"Response: {response}")

    # Example 3: With temperature control
    print("\n3. With Temperature Control (creative):")
    print("-" * 40)
    response = chat("Write a creative tagline for a coffee shop", temperature=1.5)
    print(f"Response: {response}")

    # Example 4: Multi-turn conversation with message history
    print("\n4. Multi-turn Conversation:")
    print("-" * 40)
    messages = [
        {"role": "user", "content": "My name is Alice"},
        {"role": "assistant", "content": "Nice to meet you, Alice!"},
        {"role": "user", "content": "What's my name?"},
    ]
    response = chat(messages)
    print(f"Response: {response}")

    # Example 5: Streaming response
    print("\n5. Streaming Response:")
    print("-" * 40)
    print("Response (streamed): ", end="", flush=True)
    for chunk in chat("Count from 1 to 5", stream=True):
        print(chunk, end="", flush=True)
    print()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
