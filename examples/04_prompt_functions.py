"""Prompt-based function creation examples.

Demonstrates creating reusable functions from prompt templates.

Usage:
    python examples/04_prompt_functions.py
"""

from aix import prompt_func, common_funcs, PromptFuncs


def main():
    """Run prompt function examples."""
    print("=" * 60)
    print("AIX Prompt-Based Functions Examples")
    print("=" * 60)

    # Example 1: Simple text function
    print("\n1. Simple Text Function:")
    print("-" * 40)
    summarize = prompt_func("Summarize this text in one sentence: {text}")

    text = """
    Artificial intelligence is the simulation of human intelligence processes
    by machines, especially computer systems. These processes include learning,
    reasoning, and self-correction.
    """

    summary = summarize(text=text.strip())
    print(f"Original: {text.strip()}")
    print(f"\nSummary: {summary}")

    # Example 2: Multiple parameters
    print("\n2. Function with Multiple Parameters:")
    print("-" * 40)
    translate = prompt_func("Translate '{text}' from {source_lang} to {target_lang}")

    result = translate(
        text="Hello, how are you?", source_lang="English", target_lang="Spanish"
    )
    print(f"Translation: {result}")

    # Example 3: Structured output
    print("\n3. Structured Output (JSON):")
    print("-" * 40)
    extract_contact = prompt_func(
        "Extract contact information from: {text}",
        output_schema={"name": str, "email": str, "phone": str},
    )

    contact_text = "Contact John Smith at john.smith@example.com or call 555-1234"
    contact_info = extract_contact(text=contact_text)

    print(f"Input: {contact_text}")
    print(f"\nExtracted:")
    print(f"  Name: {contact_info.get('name')}")
    print(f"  Email: {contact_info.get('email')}")
    print(f"  Phone: {contact_info.get('phone')}")

    # Example 4: Pre-built common functions
    print("\n4. Pre-built Common Functions:")
    print("-" * 40)

    # Summarize
    text = "The quick brown fox jumps over the lazy dog. This is a common pangram."
    summary = common_funcs.summarize(text=text)
    print(f"Summarize: {summary}")

    # Extract keywords
    keywords = common_funcs.extract_keywords(
        text="Machine learning and artificial intelligence are transforming technology"
    )
    print(f"\nKeywords: {keywords}")

    # Sentiment analysis
    sentiment = common_funcs.sentiment(
        text="I absolutely love this product! It's amazing!"
    )
    print(f"\nSentiment: {sentiment}")

    # Example 5: Custom function collection
    print("\n5. Custom Function Collection:")
    print("-" * 40)

    my_funcs = PromptFuncs()

    # Add custom functions
    my_funcs.add("explain_code", "Explain what this code does in simple terms: {code}")

    my_funcs.add(
        "generate_docstring", "Generate a Python docstring for this function: {code}"
    )

    code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

    explanation = my_funcs.explain_code(code=code.strip())
    print(f"Code explanation: {explanation}")

    docstring = my_funcs.generate_docstring(code=code.strip())
    print(f"\nGenerated docstring: {docstring}")

    # Example 6: Function with specific model
    print("\n6. Function with Specific Model:")
    print("-" * 40)
    creative_writer = prompt_func(
        "Write a creative one-line story about {topic}",
        model="gpt-4o-mini",
        temperature=1.5,
    )

    story = creative_writer(topic="a time-traveling cat")
    print(f"Story: {story}")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
