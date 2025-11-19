"""Model discovery and selection example.

Demonstrates discovering, filtering, and selecting AI models.

Usage:
    python examples/06_model_discovery.py
"""

from aix import models, chat


def main():
    """Run model discovery examples."""
    print("=" * 60)
    print("AIX Model Discovery Examples")
    print("=" * 60)

    # Example 1: Discover models
    print("\n1. Discovering Models from OpenRouter:")
    print("-" * 40)
    try:
        discovered = models.discover('openrouter', verbose=True)
        print(f"\nDiscovered {len(discovered)} models")
    except Exception as e:
        print(f"Note: Discovery requires API key. Error: {e}")
        print("Continuing with local examples...")

    # Example 2: List available models
    print("\n2. Listing Available Models:")
    print("-" * 40)
    all_models = list(models)
    if all_models:
        print(f"Total models: {len(all_models)}")
        print("\nFirst 5 models:")
        for model_id in all_models[:5]:
            print(f"  - {model_id}")
    else:
        print("No models discovered yet. Run models.discover() first.")

    # Example 3: Get specific model info
    print("\n3. Getting Model Information:")
    print("-" * 40)
    if 'openai/gpt-4o' in models:
        info = models['openai/gpt-4o']
        print(f"Model ID: {info.id}")
        print(f"Provider: {info.provider}")
        print(f"Context Size: {info.context_size}")
        print(f"Cost per token: {info.cost_per_token}")
        print(f"Is Local: {info.is_local}")
    else:
        print("Model not found. Discover models first.")

    # Example 4: Filter models by provider
    print("\n4. Filtering by Provider:")
    print("-" * 40)
    openai_models = models.filter(provider='openai')
    if openai_models:
        print(f"OpenAI models: {len(openai_models)}")
        for model in openai_models[:5]:
            print(f"  - {model.id}")

    # Example 5: Filter by criteria
    print("\n5. Filtering by Multiple Criteria:")
    print("-" * 40)
    filtered = models.filter(
        min_context_size=8000,
        custom_filter=lambda m: m.cost_per_token.get('input', 0) < 0.00001
    )
    if filtered:
        print(f"Models with >8K context and low cost: {len(filtered)}")
        for model in filtered[:3]:
            print(f"  - {model.id}")
            print(f"    Context: {model.context_size}")
            print(f"    Input cost: {model.cost_per_token.get('input', 'N/A')}")

    # Example 6: Search models
    print("\n6. Searching for Models:")
    print("-" * 40)
    results = models.search('gpt-4')
    if results:
        print(f"Found {len(results)} models matching 'gpt-4':")
        for model in results[:5]:
            print(f"  - {model.id}")

    # Example 7: Get recommendations
    print("\n7. Getting Model Recommendations:")
    print("-" * 40)
    recommended = models.recommend(
        task='chat',
        max_cost_per_mtok=5.0,
        min_context_size=8000
    )
    if recommended:
        print(f"Recommended models: {len(recommended)}")
        for model in recommended[:3]:
            cost = model.cost_per_token.get('input', 0) * 1_000_000
            print(f"  - {model.id}")
            print(f"    Cost: ${cost:.2f}/M tokens")
            print(f"    Context: {model.context_size}")

    # Example 8: Use model with chat
    print("\n8. Using Model with Chat:")
    print("-" * 40)
    try:
        if models:
            # Get a cheap model
            cheap_models = models.filter(
                custom_filter=lambda m: m.cost_per_token.get('input', float('inf')) < 0.000001
            )
            if cheap_models:
                model = cheap_models[0]
                print(f"Using model: {model.id}")

                response = chat(
                    "What is 2+2?",
                    model=model.id
                )
                print(f"Response: {response}")
    except Exception as e:
        print(f"Chat requires API keys: {e}")

    # Example 9: Compare models
    print("\n9. Comparing Models:")
    print("-" * 40)
    if models:
        comparison_models = list(models)[:3]
        print(f"Comparing {len(comparison_models)} models:\n")

        for model in comparison_models:
            print(f"{model.id}:")
            print(f"  Provider: {model.provider}")
            print(f"  Context: {model.context_size}")
            print(f"  Cost (input): {model.cost_per_token.get('input', 'N/A')}")
            print(f"  Local: {model.is_local}")
            print()

    # Example 10: Model by task
    print("\n10. Models by Task:")
    print("-" * 40)
    chat_models = models.by_task('chat')
    if chat_models:
        print(f"Chat models: {len(chat_models)}")
        for model in chat_models[:5]:
            print(f"  - {model.id}")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nNote: Some examples require discovering models first:")
    print("  models.discover('openrouter')  # Requires OPENROUTER_API_KEY")
    print("=" * 60)


if __name__ == "__main__":
    main()
