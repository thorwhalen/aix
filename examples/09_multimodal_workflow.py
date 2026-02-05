"""Multimodal workflow example.

Demonstrates combining multiple AI operations in a workflow.

Usage:
    python examples/09_multimodal_workflow.py
"""

from aix import (
    chat,
    generate_image,
    text_to_speech,
    transcribe,
    prompt_func,
)


def main():
    """Run multimodal workflow examples."""
    print("=" * 60)
    print("AIX Multimodal Workflow Examples")
    print("=" * 60)

    # Example 1: Text → Image → Description
    print("\n1. Text → Image → Description Workflow:")
    print("-" * 40)
    original_prompt = "A futuristic city with flying cars"
    print(f"Step 1: Generate image from: '{original_prompt}'")

    try:
        # Generate image
        image = generate_image(original_prompt)
        image.save("workflow_city.png")
        print("✓ Image generated and saved")

        # Use chat to analyze the image description
        print("\nStep 2: Create description of what was generated")
        description_prompt = f"""
        I requested an AI to generate an image with this prompt:
        "{original_prompt}"

        Describe what elements you would expect to see in this image.
        """

        description = chat(description_prompt, model="gpt-4o-mini")
        print(f"✓ Expected elements: {description[:200]}...")

    except Exception as e:
        print(f"⚠ Error: {e}")

    # Example 2: Text → Speech → Transcription → Analysis
    print("\n2. Text → Speech → Transcription → Analysis:")
    print("-" * 40)
    original_text = "Artificial intelligence is transforming how we work and live."
    print(f"Original text: {original_text}")

    try:
        # Convert to speech
        print("\nStep 1: Converting to speech...")
        audio = text_to_speech(original_text, voice="nova")
        audio.save("workflow_speech.mp3")
        print("✓ Audio generated")

        # Transcribe back
        print("\nStep 2: Transcribing audio...")
        transcribed = transcribe("workflow_speech.mp3")
        print(f"✓ Transcribed: {transcribed}")

        # Analyze the text
        print("\nStep 3: Analyzing sentiment...")
        analyze = prompt_func(
            "Analyze the sentiment and main topic of: {text}",
            output_schema={"sentiment": str, "topic": str, "confidence": float},
        )

        analysis = analyze(text=transcribed)
        print(f"✓ Analysis:")
        print(f"  Sentiment: {analysis['sentiment']}")
        print(f"  Topic: {analysis['topic']}")
        print(f"  Confidence: {analysis['confidence']}")

    except Exception as e:
        print(f"⚠ Error: {e}")

    # Example 3: Story Generation → Image → Narration
    print("\n3. Story Generation → Image → Narration:")
    print("-" * 40)

    try:
        # Generate a short story
        print("Step 1: Generating a short story...")
        story = chat(
            "Write a one-paragraph story about a robot discovering art. "
            "Keep it under 50 words.",
            model="gpt-4o-mini",
        )
        print(f"✓ Story: {story}")

        # Extract key visual element
        print("\nStep 2: Extracting visual element...")
        extract_visual = prompt_func(
            "Extract a single vivid visual scene from this story for image generation: {story}"
        )
        visual_prompt = extract_visual(story=story)
        print(f"✓ Visual prompt: {visual_prompt}")

        # Generate image
        print("\nStep 3: Generating image...")
        image = generate_image(visual_prompt)
        image.save("workflow_story_image.png")
        print("✓ Image generated")

        # Create narration
        print("\nStep 4: Creating narration...")
        audio = text_to_speech(story, voice="fable")
        audio.save("workflow_story_narration.mp3")
        print("✓ Narration created")

        print("\n✓ Complete multimedia story created:")
        print("  - Text: story.txt (in memory)")
        print("  - Image: workflow_story_image.png")
        print("  - Audio: workflow_story_narration.mp3")

    except Exception as e:
        print(f"⚠ Error: {e}")

    # Example 4: Multilingual Workflow
    print("\n4. Multilingual Workflow:")
    print("-" * 40)

    try:
        # Original English text
        english_text = "Welcome to the future of artificial intelligence."
        print(f"Original (English): {english_text}")

        # Translate to French
        print("\nStep 1: Translating to French...")
        translate = prompt_func("Translate to French: {text}")
        french_text = translate(text=english_text)
        print(f"✓ French: {french_text}")

        # Generate French speech
        print("\nStep 2: Generating French speech...")
        french_audio = text_to_speech(french_text, voice="shimmer")
        french_audio.save("workflow_french.mp3")
        print("✓ French audio generated")

        # Transcribe and translate back
        print("\nStep 3: Transcribing French audio...")
        transcribed_french = transcribe("workflow_french.mp3", language="fr")
        print(f"✓ Transcribed: {transcribed_french}")

        print("\nStep 4: Translating back to English...")
        back_to_english = translate_text = prompt_func("Translate to English: {text}")
        final_english = back_to_english(text=transcribed_french)
        print(f"✓ Back to English: {final_english}")

    except Exception as e:
        print(f"⚠ Error: {e}")

    # Example 5: Content Pipeline
    print("\n5. Content Creation Pipeline:")
    print("-" * 40)
    topic = "quantum computing"
    print(f"Topic: {topic}")

    try:
        # Generate educational content
        print("\nStep 1: Generating explanation...")
        explain = prompt_func(
            "Explain {topic} in simple terms. One paragraph, under 100 words."
        )
        explanation = explain(topic=topic)
        print(f"✓ Explanation: {explanation[:100]}...")

        # Create visual
        print("\nStep 2: Creating visual representation...")
        visual_prompt = f"Educational diagram showing {topic}"
        image = generate_image(visual_prompt)
        image.save(f"workflow_{topic.replace(' ', '_')}.png")
        print("✓ Visual created")

        # Create audio version
        print("\nStep 3: Creating audio version...")
        audio = text_to_speech(explanation, voice="onyx")
        audio.save(f"workflow_{topic.replace(' ', '_')}.mp3")
        print("✓ Audio version created")

        # Create summary
        print("\nStep 4: Creating summary...")
        summarize = prompt_func("Summarize in one sentence: {text}")
        summary = summarize(text=explanation)
        print(f"✓ Summary: {summary}")

        print("\n✓ Complete educational package created!")

    except Exception as e:
        print(f"⚠ Error: {e}")

    print("\n" + "=" * 60)
    print("Multimodal workflows completed!")
    print("=" * 60)
    print("\nThese examples show how to combine:")
    print("  - Text generation (chat, prompt_func)")
    print("  - Image generation (generate_image)")
    print("  - Audio synthesis (text_to_speech)")
    print("  - Audio transcription (transcribe)")
    print("\ninto powerful multimodal AI workflows!")
    print("=" * 60)


if __name__ == "__main__":
    main()
