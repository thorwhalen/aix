"""Stateful chat session example.

Demonstrates maintaining conversation history with ChatSession.

Usage:
    python examples/02_stateful_chat.py
"""

from aix import chat_with_history


def main():
    """Run stateful chat examples."""
    print("=" * 60)
    print("AIX Stateful Chat Session Example")
    print("=" * 60)

    # Create a chat session with system prompt
    print("\nCreating chat session with system prompt...")
    session = chat_with_history("You are a helpful math tutor. Keep answers concise.")

    # Have a multi-turn conversation
    print("\n" + "=" * 60)
    print("Conversation:")
    print("=" * 60)

    # Turn 1
    print("\nUser: What is 2+2?")
    response = session.send("What is 2+2?")
    print(f"Assistant: {response}")

    # Turn 2
    print("\nUser: And if I multiply that by 3?")
    response = session.send("And if I multiply that by 3?")
    print(f"Assistant: {response}")

    # Turn 3
    print("\nUser: What was my first question?")
    response = session.send("What was my first question?")
    print(f"Assistant: {response}")

    # Show conversation history
    print("\n" + "=" * 60)
    print(f"Total messages in history: {len(session.history)}")
    print("=" * 60)

    for i, msg in enumerate(session.history):
        role = msg["role"].capitalize()
        content = (
            msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
        )
        print(f"{i+1}. {role}: {content}")

    # Clear history and start fresh
    print("\n" + "=" * 60)
    print("Clearing history...")
    session.clear_history(keep_system=True)
    print(f"Messages after clear: {len(session.history)}")
    print("=" * 60)

    # New conversation
    print("\nUser: What is 10-5?")
    response = session.send("What is 10-5?")
    print(f"Assistant: {response}")

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
