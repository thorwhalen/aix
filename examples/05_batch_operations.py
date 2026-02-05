"""Batch operations example.

Demonstrates efficient batch processing of multiple requests.

Usage:
    python examples/05_batch_operations.py
"""

from aix import batch_chat, batch_embeddings, batch_process, BatchProcessor


def main():
    """Run batch operations examples."""
    print("=" * 60)
    print("AIX Batch Operations Examples")
    print("=" * 60)

    # Example 1: Batch chat
    print("\n1. Batch Chat:")
    print("-" * 40)
    prompts = [
        "What is 2+2?",
        "What is 3+3?",
        "What is 5+5?",
        "What is 7+7?",
        "What is 10+10?",
    ]

    print(f"Processing {len(prompts)} prompts in batch...")
    results = list(batch_chat(prompts, batch_size=5, max_workers=3, show_progress=True))

    print("\nResults:")
    for i, (prompt, result) in enumerate(zip(prompts, results), 1):
        print(f"{i}. Q: {prompt}")
        print(f"   A: {result}")

    # Example 2: Batch embeddings
    print("\n2. Batch Embeddings:")
    print("-" * 40)
    texts = [
        "machine learning",
        "deep learning",
        "neural networks",
        "artificial intelligence",
        "natural language processing",
    ] * 4  # 20 texts total

    print(f"Generating embeddings for {len(texts)} texts...")
    vectors = list(batch_embeddings(texts, batch_size=10, show_progress=True))

    print(f"\nGenerated {len(vectors)} vectors")
    print(f"Vector dimension: {len(vectors[0])}")

    # Example 3: Generic batch processing
    print("\n3. Generic Batch Processing:")
    print("-" * 40)

    from aix import chat

    def analyze_sentiment(text):
        """Analyze sentiment of text."""
        return chat(
            f"Analyze the sentiment (positive/negative/neutral): '{text}'",
            model="gpt-4o-mini",
        )

    reviews = [
        "This product is amazing! I love it!",
        "Terrible quality, very disappointed",
        "It's okay, nothing special",
        "Best purchase I've ever made!",
        "Complete waste of money",
    ]

    print(f"Analyzing sentiment for {len(reviews)} reviews...")
    sentiments = list(
        batch_process(reviews, analyze_sentiment, batch_size=3, show_progress=True)
    )

    print("\nSentiment Analysis Results:")
    for review, sentiment in zip(reviews, sentiments):
        print(f"\nReview: {review}")
        print(f"Sentiment: {sentiment}")

    # Example 4: Using BatchProcessor
    print("\n4. Using BatchProcessor:")
    print("-" * 40)

    processor = BatchProcessor(batch_size=5, max_workers=3, show_progress=True)

    questions = [
        "What is Python?",
        "What is JavaScript?",
        "What is Java?",
    ]

    print(f"Processing {len(questions)} questions...")
    results = processor.process_chats(questions, model="gpt-4o-mini")

    print("\nResults:")
    for q, r in zip(questions, results):
        print(f"\nQ: {q}")
        print(f"A: {r[:100]}...")  # First 100 chars

    # Save results to file
    print("\nSaving results to batch_results.json...")
    processor.save_results("batch_results.json")
    print("Saved!")

    # Example 5: Batch with retry logic
    print("\n5. Batch Processing with Retries:")
    print("-" * 40)

    def potentially_failing_operation(item):
        """Simulated operation that might fail."""
        # In reality, this would be an API call or similar
        return chat(f"Process: {item}", model="gpt-4o-mini")

    items = ["item1", "item2", "item3"]

    print(f"Processing {len(items)} items with retry logic...")
    results = list(
        batch_process(
            items,
            potentially_failing_operation,
            retry_attempts=3,
            retry_delay=1.0,
            show_progress=True,
        )
    )

    print(f"\nProcessed {len(results)} items successfully")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
