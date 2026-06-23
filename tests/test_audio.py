"""Tests for aix.audio module."""

import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
from aix.audio import (
    text_to_speech,
    transcribe,
    transcribe_with_timestamps,
    GeneratedAudio,
    TranscriptionResult,
)


class TestGeneratedAudio:
    """Tests for GeneratedAudio class."""

    def test_initialization(self):
        """Test initialization."""
        audio = GeneratedAudio(
            data=b"fake_audio_data", model="tts-1", text="Hello world", voice="alloy"
        )
        assert audio.data == b"fake_audio_data"
        assert audio.model == "tts-1"
        assert audio.text == "Hello world"
        assert audio.voice == "alloy"

    def test_as_bytes(self):
        """Test getting audio as bytes."""
        data = b"audio_bytes"
        audio = GeneratedAudio(data=data)
        assert audio.as_bytes() == data

    @patch("builtins.open", new_callable=mock_open)
    def test_save(self, mock_file):
        """Test saving audio to file."""
        audio = GeneratedAudio(data=b"test_audio")
        audio.save("output.mp3")

        mock_file.assert_called_once_with(Path("output.mp3"), "wb")
        mock_file().write.assert_called_once_with(b"test_audio")

    def test_repr(self):
        """Test string representation."""
        audio = GeneratedAudio(data=b"12345", voice="nova", format="mp3")
        repr_str = repr(audio)
        assert "GeneratedAudio" in repr_str
        assert "nova" in repr_str


class TestTranscriptionResult:
    """Tests for TranscriptionResult class."""

    def test_initialization(self):
        """Test initialization."""
        result = TranscriptionResult(
            text="Hello world", language="en", duration=5.0, model="whisper-1"
        )
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.duration == 5.0
        assert result.model == "whisper-1"

    def test_str(self):
        """Test string conversion."""
        result = TranscriptionResult(text="Test text")
        assert str(result) == "Test text"

    def test_repr(self):
        """Test representation."""
        result = TranscriptionResult(text="Hello", language="en")
        repr_str = repr(result)
        assert "TranscriptionResult" in repr_str
        assert "en" in repr_str


class TestTextToSpeech:
    """Tests for text_to_speech function."""

    @patch("aix.audio._litellm_speech")
    def test_simple_tts(self, mock_speech):
        """Test simple text-to-speech."""
        mock_speech.return_value = b"fake_audio_data"

        result = text_to_speech("Hello, world!")

        assert isinstance(result, GeneratedAudio)
        assert result.data == b"fake_audio_data"
        assert result.text == "Hello, world!"

        mock_speech.assert_called_once()
        call_kwargs = mock_speech.call_args[1]
        assert call_kwargs["input"] == "Hello, world!"

    @patch("aix.audio._litellm_speech")
    def test_tts_with_voice(self, mock_speech):
        """Test TTS with specific voice."""
        mock_speech.return_value = b"audio"

        text_to_speech("Test", voice="nova")

        call_kwargs = mock_speech.call_args[1]
        assert call_kwargs["voice"] == "nova"

    @patch("aix.audio._litellm_speech")
    def test_tts_with_speed(self, mock_speech):
        """Test TTS with custom speed."""
        mock_speech.return_value = b"audio"

        text_to_speech("Test", speed=1.5)

        call_kwargs = mock_speech.call_args[1]
        assert call_kwargs["speed"] == 1.5

    @patch("aix.audio._litellm_speech")
    def test_tts_with_model(self, mock_speech):
        """Test TTS with specific model."""
        mock_speech.return_value = b"audio"

        text_to_speech("Test", model="tts-1-hd")

        call_kwargs = mock_speech.call_args[1]
        assert call_kwargs["model"] == "tts-1-hd"


class TestTranscribe:
    """Tests for transcribe function."""

    @patch("aix.audio._litellm_transcription")
    @patch("builtins.open", new_callable=mock_open, read_data=b"fake_audio")
    def test_transcribe_file_path(self, mock_file, mock_transcription):
        """Test transcribing from file path."""
        mock_transcription.return_value = "Transcribed text"

        result = transcribe("audio.mp3")

        assert result == "Transcribed text"
        mock_file.assert_called_once_with(Path("audio.mp3"), "rb")

    @patch("aix.audio._litellm_transcription")
    def test_transcribe_bytes(self, mock_transcription):
        """Test transcribing from bytes."""
        mock_transcription.return_value = "Transcribed text"

        result = transcribe(b"fake_audio_data")

        assert result == "Transcribed text"

    @patch("aix.audio._litellm_transcription")
    @patch("builtins.open", new_callable=mock_open, read_data=b"audio")
    def test_transcribe_with_language(self, mock_file, mock_transcription):
        """Test transcription with language hint."""
        mock_transcription.return_value = "Text"

        transcribe("audio.mp3", language="es")

        call_kwargs = mock_transcription.call_args[1]
        assert call_kwargs["language"] == "es"

    @patch("aix.audio._litellm_transcription")
    @patch("builtins.open", new_callable=mock_open, read_data=b"audio")
    def test_transcribe_with_prompt(self, mock_file, mock_transcription):
        """Test transcription with style prompt."""
        mock_transcription.return_value = "Text"

        transcribe("audio.mp3", prompt="Technical discussion")

        call_kwargs = mock_transcription.call_args[1]
        assert call_kwargs["prompt"] == "Technical discussion"

    @patch("aix.audio._litellm_transcription")
    @patch("builtins.open", new_callable=mock_open, read_data=b"audio")
    def test_transcribe_verbose_json(self, mock_file, mock_transcription):
        """Test transcription with verbose JSON format."""
        # Mock verbose response
        mock_response = Mock()
        mock_response.text = "Transcribed text"
        mock_response.language = "en"
        mock_response.duration = 10.5
        mock_response.segments = [
            {"start": 0.0, "end": 5.0, "text": "First part"},
            {"start": 5.0, "end": 10.5, "text": "Second part"},
        ]
        mock_transcription.return_value = mock_response

        result = transcribe("audio.mp3", response_format="verbose_json")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Transcribed text"
        assert result.language == "en"
        assert result.duration == 10.5
        assert len(result.segments) == 2


class TestTranscribeWithTimestamps:
    """Tests for transcribe_with_timestamps function."""

    @patch("aix.audio.transcribe")
    def test_with_timestamps(self, mock_transcribe):
        """Test transcription with timestamps."""
        mock_result = TranscriptionResult(
            text="Test", segments=[{"start": 0, "end": 1, "text": "Test"}]
        )
        mock_transcribe.return_value = mock_result

        result = transcribe_with_timestamps("audio.mp3")

        assert isinstance(result, TranscriptionResult)
        mock_transcribe.assert_called_once()
        call_kwargs = mock_transcribe.call_args[1]
        assert call_kwargs["response_format"] == "verbose_json"
        assert "segment" in call_kwargs["timestamp_granularities"]

    @patch("aix.audio.transcribe")
    def test_word_granularity(self, mock_transcribe):
        """Test word-level timestamps."""
        mock_result = TranscriptionResult(text="Test")
        mock_transcribe.return_value = mock_result

        transcribe_with_timestamps("audio.mp3", granularity="word")

        call_kwargs = mock_transcribe.call_args[1]
        assert "word" in call_kwargs["timestamp_granularities"]


def _fake_scribed_module():
    """A stand-in `scribed` module whose transcribe() returns a fake Transcript."""
    import types

    class _Seg:
        def __init__(self, start, end, text, speaker=None, confidence=None):
            self.start = start
            self.end = end
            self.text = text
            self.speaker = speaker
            self.confidence = confidence

    class _Transcript:
        text = "Hello world."
        language = "en"
        duration = 1.5
        srt = "1\n00:00:00,000 --> 00:00:01,500\nHello world.\n"
        vtt = "WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nHello world.\n"
        segments = [_Seg(0.0, 1.5, "Hello world.", "A", 0.9)]

    mod = types.ModuleType("scribed")
    mod._last_call = {}

    def transcribe_(audio, *, backend=None, **kwargs):
        mod._last_call = {"audio": audio, "backend": backend, "kwargs": kwargs}
        return _Transcript()

    mod.transcribe = transcribe_
    return mod


class TestTranscribeScribedDelegation:
    """The non-breaking engine= delegation to the `scribed` package."""

    def test_engine_text_returns_str(self):
        import sys

        fake = _fake_scribed_module()
        with patch.dict(sys.modules, {"scribed": fake}):
            out = transcribe("a.wav", engine="faster-whisper")
        assert out == "Hello world."
        assert fake._last_call["backend"] == "faster-whisper"

    def test_engine_srt_and_vtt(self):
        import sys

        fake = _fake_scribed_module()
        with patch.dict(sys.modules, {"scribed": fake}):
            srt = transcribe("a.wav", engine="deepgram", response_format="srt")
            vtt = transcribe("a.wav", engine="deepgram", response_format="vtt")
        assert srt.startswith("1\n")
        assert vtt.startswith("WEBVTT")

    def test_engine_verbose_returns_transcription_result(self):
        import sys

        fake = _fake_scribed_module()
        with patch.dict(sys.modules, {"scribed": fake}):
            res = transcribe(
                "a.wav",
                engine="faster-whisper",
                language="en",
                diarize=True,
                response_format="verbose_json",
            )
        assert isinstance(res, TranscriptionResult)
        assert res.text == "Hello world."
        assert res.language == "en"
        assert res.model == "scribed:faster-whisper"
        assert res.segments[0]["speaker"] == "A"
        assert res.segments[0]["start"] == 0.0
        # extra kwargs (diarize) and language flow through to scribed
        assert fake._last_call["kwargs"].get("diarize") is True
        assert fake._last_call["kwargs"].get("language") == "en"

    @patch("aix.audio._transcribe_litellm")
    def test_no_engine_uses_litellm(self, mock_litellm):
        """Without engine=, behavior is unchanged (LiteLLM path)."""
        mock_litellm.return_value = "litellm text"
        out = transcribe("a.wav")
        assert out == "litellm text"
        mock_litellm.assert_called_once()

    def test_missing_scribed_raises_helpful_error(self):
        import sys

        # Mapping the name to None makes `import scribed` raise ImportError.
        with patch.dict(sys.modules, {"scribed": None}):
            with pytest.raises(ImportError, match="aix\\[scribed\\]"):
                transcribe("a.wav", engine="faster-whisper")
