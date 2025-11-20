"""Image generation examples.

Demonstrates text-to-image generation with various options.

Usage:
    python examples/07_image_generation.py
"""

from aix import generate_image, generate_images


def main():
    """Run image generation examples."""
    print("=" * 60)
    print("AIX Image Generation Examples")
    print("=" * 60)

    # Example 1: Simple image generation
    print("\n1. Simple Image Generation:")
    print("-" * 40)
    print("Generating: 'A serene mountain landscape at sunset'")
    try:
        image = generate_image("A serene mountain landscape at sunset")
        image.save("mountain_sunset.png")
        print(f"✓ Saved to: mountain_sunset.png")
        print(f"  Model: {image.model}")
        if image.revised_prompt:
            print(f"  Revised prompt: {image.revised_prompt}")
    except Exception as e:
        print(f"⚠ Error: {e}")
        print("  (Requires API key, e.g., OPENAI_API_KEY)")

    # Example 2: High quality with DALL-E 3
    print("\n2. High Quality (DALL-E 3):")
    print("-" * 40)
    print("Generating: 'Abstract art with vibrant colors'")
    try:
        image = generate_image(
            "Abstract art with vibrant colors",
            model="dall-e-3",
            quality="hd",
            style="vivid"
        )
        image.save("abstract_art_hd.png")
        print(f"✓ Saved to: abstract_art_hd.png")
        print(f"  Quality: HD")
        print(f"  Style: Vivid")
    except Exception as e:
        print(f"⚠ Error: {e}")

    # Example 3: Different sizes
    print("\n3. Different Image Sizes:")
    print("-" * 40)
    sizes = ["256x256", "512x512", "1024x1024"]
    for size in sizes:
        print(f"\nGenerating {size} image...")
        try:
            image = generate_image(
                "A cute robot waving hello",
                size=size,
                model="dall-e-2"  # DALL-E 2 supports multiple sizes
            )
            filename = f"robot_{size}.png"
            image.save(filename)
            print(f"✓ Saved to: {filename}")
        except Exception as e:
            print(f"⚠ Error for {size}: {e}")

    # Example 4: Generate multiple variations
    print("\n4. Multiple Variations:")
    print("-" * 40)
    print("Generating 3 variations of: 'A magical forest'")
    try:
        images = generate_images(
            "A magical forest with glowing mushrooms",
            n=3,
            model="dall-e-2"
        )
        print(f"✓ Generated {len(images)} images")
        for i, img in enumerate(images):
            filename = f"forest_variation_{i+1}.png"
            img.save(filename)
            print(f"  Saved variation {i+1} to: {filename}")
    except Exception as e:
        print(f"⚠ Error: {e}")

    # Example 5: Different styles
    print("\n5. Different Styles (DALL-E 3):")
    print("-" * 40)
    prompt = "A coffee shop interior"
    styles = ["natural", "vivid"]

    for style in styles:
        print(f"\nGenerating with '{style}' style...")
        try:
            image = generate_image(
                prompt,
                model="dall-e-3",
                style=style
            )
            filename = f"coffee_shop_{style}.png"
            image.save(filename)
            print(f"✓ Saved to: {filename}")
        except Exception as e:
            print(f"⚠ Error: {e}")

    # Example 6: Creative prompts
    print("\n6. Creative Prompts:")
    print("-" * 40)
    creative_prompts = [
        "A steampunk airship flying through clouds",
        "A cozy reading nook with bookshelves and a window",
        "A neon-lit cyberpunk street at night",
    ]

    for i, prompt in enumerate(creative_prompts, 1):
        print(f"\n{i}. {prompt}")
        try:
            image = generate_image(prompt)
            filename = f"creative_{i}.png"
            image.save(filename)
            print(f"   ✓ Saved to: {filename}")
        except Exception as e:
            print(f"   ⚠ Error: {e}")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nNote: Image generation requires API keys (e.g., OPENAI_API_KEY)")
    print("Set your key: export OPENAI_API_KEY=your-key-here")
    print("=" * 60)


if __name__ == "__main__":
    main()
