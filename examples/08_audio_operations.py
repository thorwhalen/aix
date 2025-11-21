"""Audio operations examples.

Demonstrates text-to-speech and speech-to-text capabilities.

Usage:
    python examples/08_audio_operations.py
"""

from aix import text_to_speech, transcribe, transcribe_with_timestamps


def main():
    """Run audio operation examples."""
    print("=" * 60)
    print("AIX Audio Operations Examples")
    print("=" * 60)

    # Example 1: Simple text-to-speech
    print("\n1. Simple Text-to-Speech:")
    print("-" * 40)
    text = "Hello! This is a test of the text-to-speech system."
    print(f"Text: {text}")

    try:
        audio = text_to_speech(text)
        audio.save("hello_tts.mp3")
        print(f"✓ Generated audio saved to: hello_tts.mp3")
        print(f"  Model: {audio.model}")
        print(f"  Voice: {audio.voice}")
        print(f"  Format: {audio.format}")
    except Exception as e:
        print(f"⚠ Error: {e}")
        print("  (Requires API key, e.g., OPENAI_API_KEY)")

    # Example 2: Different voices
    print("\n2. Different Voices:")
    print("-" * 40)
    voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    text = "The quick brown fox jumps over the lazy dog."

    for voice in voices:
        print(f"\nGenerating with voice: {voice}")
        try:
            audio = text_to_speech(text, voice=voice)
            filename = f"voice_{voice}.mp3"
            audio.save(filename)
            print(f"✓ Saved to: {filename}")
        except Exception as e:
            print(f"⚠ Error: {e}")

    # Example 3: Speed variations
    print("\n3. Speed Variations:")
    print("-" * 40)
    text = "This is a speed test."
    speeds = [0.5, 1.0, 1.5, 2.0]

    for speed in speeds:
        print(f"\nGenerating at {speed}x speed...")
        try:
            audio = text_to_speech(text, speed=speed)
            filename = f"speed_{speed:.1f}x.mp3"
            audio.save(filename)
            print(f"✓ Saved to: {filename}")
        except Exception as e:
            print(f"⚠ Error: {e}")

    # Example 4: High quality TTS
    print("\n4. High Quality TTS:")
    print("-" * 40)
    text = "This is high quality text-to-speech audio."
    print(f"Text: {text}")

    try:
        audio = text_to_speech(
            text,
            model="tts-1-hd",
            voice="nova"
        )
        audio.save("high_quality_tts.mp3")
        print(f"✓ Generated with tts-1-hd model")
        print(f"  Saved to: high_quality_tts.mp3")
    except Exception as e:
        print(f"⚠ Error: {e}")

    # Example 5: Transcribe audio file
    print("\n5. Transcribe Audio File:")
    print("-" * 40)
    print("Transcribing: recording.mp3")

    try:
        text = transcribe("recording.mp3")
        print(f"✓ Transcription: {text}")
    except FileNotFoundError:
        print("⚠ File 'recording.mp3' not found")
        print("  Create a test audio file or use text-to-speech first:")
        try:
            test_audio = text_to_speech("This is a test recording for transcription.")
            test_audio.save("test_recording.mp3")
            print("  ✓ Created test_recording.mp3")

            # Now transcribe it
            result = transcribe("test_recording.mp3")
            print(f"  ✓ Transcription: {result}")
        except Exception as e:
            print(f"  ⚠ Error: {e}")
    except Exception as e:
        print(f"⚠ Error: {e}")

    # Example 6: Transcribe with language hint
    print("\n6. Transcribe with Language Hint:")
    print("-" * 40)
    print("Transcribing with language='en'")

    try:
        text = transcribe("test_recording.mp3", language="en")
        print(f"✓ Transcription: {text}")
    except Exception as e:
        print(f"⚠ Error: {e}")

    # Example 7: Detailed transcription with timestamps
    print("\n7. Transcription with Timestamps:")
    print("-" * 40)

    try:
        result = transcribe_with_timestamps("test_recording.mp3")
        print(f"✓ Text: {result.text}")
        print(f"  Language: {result.language}")
        print(f"  Duration: {result.duration}s")

        if result.segments:
            print("\n  Segments:")
            for seg in result.segments[:3]:  # Show first 3
                start = seg.get('start', 0)
                end = seg.get('end', 0)
                text = seg.get('text', '')
                print(f"    [{start:.2f}-{end:.2f}] {text}")
    except Exception as e:
        print(f"⚠ Error: {e}")

    # Example 8: Round-trip (TTS + Transcription)
    print("\n8. Round-trip Test (TTS → Transcription):")
    print("-" * 40)
    original_text = "The quick brown fox jumps over the lazy dog."
    print(f"Original: {original_text}")

    try:
        # Generate audio
        audio = text_to_speech(original_text)
        audio.save("roundtrip.mp3")
        print("✓ Generated audio")

        # Transcribe it back
        transcribed_text = transcribe("roundtrip.mp3")
        print(f"✓ Transcribed: {transcribed_text}")

        # Compare
        if original_text.lower() in transcribed_text.lower():
            print("✓ Round-trip successful!")
        else:
            print("⚠ Transcription differs from original")
    except Exception as e:
        print(f"⚠ Error: {e}")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nNote: Audio operations require API keys (e.g., OPENAI_API_KEY)")
    print("Set your key: export OPENAI_API_KEY=your-key-here")
    print("=" * 60)


if __name__ == "__main__":
    main()
