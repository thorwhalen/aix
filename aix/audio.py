"""Audio operations interface for AIX.

Provides text-to-speech (TTS) and speech-to-text (transcription) capabilities.

Examples:
    Text to speech:
    >>> from aix.audio import text_to_speech
    >>> audio = text_to_speech("Hello, world!")  # doctest: +SKIP
    >>> audio.save("hello.mp3")  # doctest: +SKIP

    Transcription:
    >>> from aix.audio import transcribe
    >>> text = transcribe("speech.mp3")  # doctest: +SKIP
    >>> print(text)  # doctest: +SKIP
    'Hello, this is a test recording.'
"""

from collections.abc import Iterable
from typing import Union, BinaryIO
from pathlib import Path
import mimetypes

# Import LiteLLM but keep it private
try:
    from litellm import transcription as _litellm_transcription
    from litellm import speech as _litellm_speech
except ImportError:
    _litellm_transcription = None
    _litellm_speech = None


# Shipped-default constants, kept for backward compatibility. The *active*
# defaults are resolved from ``aix.config`` at call time (see aix/config.py).
from aix.config import get_config as _get_config, AudioConfig as _AudioConfig

DFLT_TTS_MODEL = _AudioConfig().tts_model
DFLT_TTS_VOICE = _AudioConfig().tts_voice  # alloy, echo, fable, onyx, nova, shimmer
DFLT_TTS_SPEED = _AudioConfig().tts_speed
DFLT_TRANSCRIPTION_MODEL = _AudioConfig().transcription_model


class GeneratedAudio:
    """Wrapper for generated audio.

    Provides convenient access to audio data and saving.

    Examples:
        >>> audio = GeneratedAudio(data=b'...', model="tts-1")  # doctest: +SKIP
        >>> audio.save("output.mp3")  # doctest: +SKIP
        >>> data = audio.as_bytes()  # doctest: +SKIP
    """

    def __init__(
        self,
        data: bytes,
        model: str = None,
        text: str = None,
        voice: str = None,
        format: str = "mp3",
    ):
        """Initialize generated audio.

        Args:
            data: Audio data as bytes
            model: Model used for generation
            text: Original text
            voice: Voice used
            format: Audio format (mp3, opus, aac, flac)
        """
        self.data = data
        self.model = model
        self.text = text
        self.voice = voice
        self.format = format

    def as_bytes(self) -> bytes:
        """Get audio as bytes.

        Returns:
            Audio data as bytes
        """
        return self.data

    def save(self, path: Union[str, Path]):
        """Save audio to file.

        Args:
            path: Output file path

        Examples:
            >>> audio.save("output.mp3")  # doctest: +SKIP
            >>> audio.save("speech.wav")  # doctest: +SKIP
        """
        path = Path(path)
        with open(path, "wb") as f:
            f.write(self.data)

    def play(self):
        """Play the audio.

        Requires a system audio player or library like pygame/pyaudio.

        Examples:
            >>> audio.play()  # doctest: +SKIP
        """
        # Try to play using system tools
        import subprocess
        import tempfile

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=f".{self.format}", delete=False) as f:
            temp_path = f.name
            f.write(self.data)

        try:
            # Try different players based on platform
            import sys

            if sys.platform == "darwin":  # macOS
                subprocess.run(["afplay", temp_path])
            elif sys.platform == "linux":
                subprocess.run(["aplay", temp_path])
            elif sys.platform == "win32":  # Windows
                import os

                os.startfile(temp_path)
        finally:
            # Clean up temp file
            import os

            try:
                os.unlink(temp_path)
            except:
                pass

    def __repr__(self) -> str:
        """String representation."""
        size_kb = len(self.data) / 1024
        return f"GeneratedAudio(format={self.format}, size={size_kb:.1f}KB, voice={self.voice})"


class TranscriptionResult:
    """Result of audio transcription.

    Contains the transcribed text and optional metadata like segments and timestamps.

    Examples:
        >>> result = TranscriptionResult(text="Hello world")  # doctest: +SKIP
        >>> print(result.text)  # doctest: +SKIP
        'Hello world'
    """

    def __init__(
        self,
        text: str,
        language: str = None,
        duration: float = None,
        segments: list = None,
        model: str = None,
    ):
        """Initialize transcription result.

        Args:
            text: Transcribed text
            language: Detected language
            duration: Audio duration in seconds
            segments: List of transcript segments with timestamps
            model: Model used for transcription
        """
        self.text = text
        self.language = language
        self.duration = duration
        self.segments = segments or []
        self.model = model

    def __str__(self) -> str:
        """Get text representation."""
        return self.text

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TranscriptionResult(text='{self.text[:50]}...', language={self.language})"
        )


def text_to_speech(
    text: str,
    *,
    model: str = None,
    voice: str = None,
    speed: float = None,
    response_format: str = "mp3",
    **kwargs,
) -> GeneratedAudio:
    """Convert text to speech audio.

    Args:
        text: Text to convert to speech
        model: TTS model to use (e.g., 'tts-1', 'tts-1-hd')
        voice: Voice to use ('alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer')
        speed: Playback speed (0.25 to 4.0)
        response_format: Audio format ('mp3', 'opus', 'aac', 'flac')
        **kwargs: Additional provider-specific parameters

    Returns:
        GeneratedAudio object

    Raises:
        ImportError: If LiteLLM is not installed

    Examples:
        >>> from aix.audio import text_to_speech
        >>> audio = text_to_speech("Hello, how are you?")  # doctest: +SKIP
        >>> audio.save("greeting.mp3")  # doctest: +SKIP

        >>> # Different voice and speed
        >>> audio = text_to_speech(
        ...     "This is a test.",
        ...     voice="nova",
        ...     speed=1.2
        ... )  # doctest: +SKIP

        >>> # High quality
        >>> audio = text_to_speech(
        ...     "Important announcement",
        ...     model="tts-1-hd",
        ...     voice="onyx"
        ... )  # doctest: +SKIP
    """
    if _litellm_speech is None:
        raise ImportError(
            "LiteLLM is required for text-to-speech. "
            "Install it with: pip install litellm"
        )

    # Apply defaults from the active config (explicit args still win)
    _audio_cfg = _get_config().audio
    model = model or _audio_cfg.tts_model
    voice = voice or _audio_cfg.tts_voice
    speed = speed if speed is not None else _audio_cfg.tts_speed

    # Build parameters
    params = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": response_format,
        "speed": speed,
    }

    # Add additional kwargs
    params.update(kwargs)

    # Call LiteLLM
    response = _litellm_speech(**params)

    # Response is audio data (bytes)
    if isinstance(response, bytes):
        audio_data = response
    else:
        # Some responses might be objects with content
        audio_data = getattr(response, "content", response)

    return GeneratedAudio(
        data=audio_data, model=model, text=text, voice=voice, format=response_format
    )


def transcribe(
    audio: Union[str, Path, BinaryIO, bytes],
    *,
    model: str = None,
    language: str = None,
    prompt: str = None,
    response_format: str = "text",
    temperature: float = None,
    timestamp_granularities: list[str] = None,
    **kwargs,
) -> Union[str, TranscriptionResult]:
    """Transcribe audio to text.

    Args:
        audio: Audio file path, file object, or bytes
        model: Transcription model (e.g., 'whisper-1')
        language: Source language (ISO-639-1 code, e.g., 'en', 'es')
        prompt: Optional text to guide the model's style
        response_format: Format of response ('text', 'json', 'verbose_json', 'srt', 'vtt')
        temperature: Sampling temperature (0 to 1)
        timestamp_granularities: List of timestamp types ('word', 'segment')
        **kwargs: Additional provider-specific parameters

    Returns:
        If response_format='text': String with transcription
        Otherwise: TranscriptionResult object with metadata

    Raises:
        ImportError: If LiteLLM is not installed

    Examples:
        >>> from aix.audio import transcribe
        >>> text = transcribe("recording.mp3")  # doctest: +SKIP
        >>> print(text)  # doctest: +SKIP
        'This is the transcribed text.'

        >>> # With language hint
        >>> text = transcribe("spanish.mp3", language="es")  # doctest: +SKIP

        >>> # Get detailed results
        >>> result = transcribe(
        ...     "meeting.mp3",
        ...     response_format="verbose_json"
        ... )  # doctest: +SKIP
        >>> print(result.text)  # doctest: +SKIP
        >>> print(result.language)  # doctest: +SKIP
        >>> for segment in result.segments:  # doctest: +SKIP
        ...     print(f"{segment['start']}: {segment['text']}")
    """
    if _litellm_transcription is None:
        raise ImportError(
            "LiteLLM is required for transcription. "
            "Install it with: pip install litellm"
        )

    # Handle different audio input types
    if isinstance(audio, (str, Path)):
        # File path
        audio_path = Path(audio)
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        filename = audio_path.name
    elif isinstance(audio, bytes):
        # Raw bytes
        audio_data = audio
        filename = "audio.mp3"  # Default filename
    else:
        # File-like object
        audio_data = audio.read()
        filename = getattr(audio, "name", "audio.mp3")

    # Apply defaults
    model = model or _get_config().audio.transcription_model

    # Build parameters
    params = {
        "model": model,
        "file": (filename, audio_data),
        "response_format": response_format,
    }

    if language:
        params["language"] = language
    if prompt:
        params["prompt"] = prompt
    if temperature is not None:
        params["temperature"] = temperature
    if timestamp_granularities:
        params["timestamp_granularities"] = timestamp_granularities

    # Add additional kwargs
    params.update(kwargs)

    # Call LiteLLM
    response = _litellm_transcription(**params)

    # Parse response based on format
    if response_format == "text":
        # Simple text response
        if isinstance(response, str):
            return response
        else:
            return getattr(response, "text", str(response))
    else:
        # Structured response
        if isinstance(response, str):
            # Fallback if we got text anyway
            return TranscriptionResult(text=response, model=model)

        # Extract metadata
        text = getattr(response, "text", "")
        language = getattr(response, "language", None)
        duration = getattr(response, "duration", None)
        segments = getattr(response, "segments", None)

        return TranscriptionResult(
            text=text,
            language=language,
            duration=duration,
            segments=segments,
            model=model,
        )


def transcribe_with_timestamps(
    audio: Union[str, Path, BinaryIO, bytes],
    *,
    granularity: str = "segment",
    model: str = None,
    **kwargs,
) -> TranscriptionResult:
    """Transcribe audio with detailed timestamps.

    Args:
        audio: Audio file path, file object, or bytes
        granularity: Timestamp granularity ('word' or 'segment')
        model: Transcription model
        **kwargs: Additional parameters for transcribe()

    Returns:
        TranscriptionResult with detailed segments

    Examples:
        >>> from aix.audio import transcribe_with_timestamps
        >>> result = transcribe_with_timestamps("lecture.mp3")  # doctest: +SKIP
        >>> for segment in result.segments:  # doctest: +SKIP
        ...     start = segment['start']
        ...     end = segment['end']
        ...     text = segment['text']
        ...     print(f"[{start:.2f}-{end:.2f}] {text}")
    """
    return transcribe(
        audio,
        model=model,
        response_format="verbose_json",
        timestamp_granularities=[granularity],
        **kwargs,
    )


def translate_audio(
    audio: Union[str, Path, BinaryIO, bytes],
    *,
    model: str = None,
    prompt: str = None,
    **kwargs,
) -> str:
    """Translate audio from any language to English.

    Note: Currently uses Whisper's translation capability which translates to English.

    Args:
        audio: Audio file path, file object, or bytes
        model: Translation model (typically 'whisper-1')
        prompt: Optional text to guide translation
        **kwargs: Additional provider-specific parameters

    Returns:
        Translated text in English

    Examples:
        >>> from aix.audio import translate_audio
        >>> english_text = translate_audio("spanish_audio.mp3")  # doctest: +SKIP
        >>> print(english_text)  # doctest: +SKIP
        'This is the English translation.'
    """
    if _litellm_transcription is None:
        raise ImportError(
            "LiteLLM is required for audio translation. "
            "Install it with: pip install litellm"
        )

    # Handle different audio input types (same as transcribe)
    if isinstance(audio, (str, Path)):
        audio_path = Path(audio)
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        filename = audio_path.name
    elif isinstance(audio, bytes):
        audio_data = audio
        filename = "audio.mp3"
    else:
        audio_data = audio.read()
        filename = getattr(audio, "name", "audio.mp3")

    model = model or _get_config().audio.transcription_model

    # Use translation endpoint
    params = {
        "model": model,
        "file": (filename, audio_data),
    }

    if prompt:
        params["prompt"] = prompt

    params.update(kwargs)

    # Note: LiteLLM may have a separate translation function
    # For now, we use transcription with task='translate'
    # This depends on provider support
    try:
        from litellm import translation as _litellm_translation

        response = _litellm_translation(**params)
    except (ImportError, AttributeError):
        # Fallback: some models support translation via transcription
        params["task"] = "translate"
        response = _litellm_transcription(**params)

    if isinstance(response, str):
        return response
    else:
        return getattr(response, "text", str(response))
