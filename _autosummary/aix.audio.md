# aix.audio

Audio operations interface for AIX.

Provides text-to-speech (TTS) and speech-to-text (transcription) capabilities.

### Examples

Text to speech:

```pycon
>>> from aix.audio import text_to_speech
>>> audio = text_to_speech("Hello, world!")
>>> audio.save("hello.mp3")
```

Transcription:

```pycon
>>> from aix.audio import transcribe
>>> text = transcribe("speech.mp3")
>>> print(text)
'Hello, this is a test recording.'
```

### Functions

| [`text_to_speech`](#aix.audio.text_to_speech)(text, \*[, model, voice, ...])    | Convert text to speech audio.                 |
|---------------------------------------------------------------------------------------------------|-----------------------------------------------|
| [`transcribe`](#aix.audio.transcribe)(audio, \*[, engine, model, ...])      | Transcribe audio to text.                     |
| [`transcribe_with_timestamps`](#aix.audio.transcribe_with_timestamps)(audio, \*[, ...])     | Transcribe audio with detailed timestamps.    |
| [`translate_audio`](#aix.audio.translate_audio)(audio, \*[, model, prompt, ...]) | Translate audio from any language to English. |

### Classes

| [`GeneratedAudio`](#aix.audio.GeneratedAudio)(data[, model, text, voice, ...])   | Wrapper for generated audio.   |
|----------------------------------------------------------------------------------------------------|--------------------------------|
| [`TranscriptionResult`](#aix.audio.TranscriptionResult)(text[, language, ...])        | Result of audio transcription. |

### *class* aix.audio.GeneratedAudio(data, model=None, text=None, voice=None, format='mp3')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Wrapper for generated audio.

Provides convenient access to audio data and saving.

### Examples

```pycon
>>> audio = GeneratedAudio(data=b'...', model="tts-1")
>>> audio.save("output.mp3")
>>> data = audio.as_bytes()
```

#### as_bytes()

Get audio as bytes.

* **Return type:**
  [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)
* **Returns:**
  Audio data as bytes

#### play()

Play the audio.

Requires a system audio player or library like pygame/pyaudio.

### Examples

```pycon
>>> audio.play()
```

#### save(path)

Save audio to file.

* **Parameters:**
  **path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Output file path

### Examples

```pycon
>>> audio.save("output.mp3")
>>> audio.save("speech.wav")
```

### *class* aix.audio.TranscriptionResult(text, language=None, duration=None, segments=None, model=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Result of audio transcription.

Contains the transcribed text and optional metadata like segments and timestamps.

### Examples

```pycon
>>> result = TranscriptionResult(text="Hello world")
>>> print(result.text)
'Hello world'
```

### aix.audio.text_to_speech(text, , model=None, voice=None, speed=None, response_format='mp3', api_key=None, \*\*kwargs)

Convert text to speech audio.

* **Parameters:**
  * **text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Text to convert to speech
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – TTS model to use (e.g., ‘tts-1’, ‘tts-1-hd’)
  * **voice** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Voice to use (‘alloy’, ‘echo’, ‘fable’, ‘onyx’, ‘nova’, ‘shimmer’)
  * **speed** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Playback speed (0.25 to 4.0)
  * **response_format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Audio format (‘mp3’, ‘opus’, ‘aac’, ‘flac’)
  * **\*\*kwargs** – Additional provider-specific parameters
* **Return type:**
  [`GeneratedAudio`](#aix.audio.GeneratedAudio)
* **Returns:**
  GeneratedAudio object
* **Raises:**
  [**ImportError**](https://docs.python.org/3/builtins/exceptions.html#ImportError) – If LiteLLM is not installed

### Examples

```pycon
>>> from aix.audio import text_to_speech
>>> audio = text_to_speech("Hello, how are you?")
>>> audio.save("greeting.mp3")
```

```pycon
>>> # Different voice and speed
>>> audio = text_to_speech(
...     "This is a test.",
...     voice="nova",
...     speed=1.2
... )
```

```pycon
>>> # High quality
>>> audio = text_to_speech(
...     "Important announcement",
...     model="tts-1-hd",
...     voice="onyx"
... )
```

### aix.audio.transcribe(audio, , engine=None, model=None, language=None, prompt=None, response_format='text', temperature=None, timestamp_granularities=None, api_key=None, \*\*kwargs)

Transcribe audio to text.

By default this routes through LiteLLM (OpenAI-style transcription). Pass
`engine=` to instead delegate to a `scribed` backend — one façade over
many ASR engines (local Whisper / faster-whisper / vosk, or cloud Deepgram /
AssemblyAI / Groq / ElevenLabs / Google …) with speaker diarization and
SRT/VTT output. The return type is unchanged either way, so existing callers
are unaffected.

* **Parameters:**
  * **audio** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`BinaryIO`](https://docs.python.org/3/library/typing.html#typing.BinaryIO), [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)]) – Audio file path, file object, or bytes.
  * **engine** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Optional `scribed` backend id (e.g. `"faster-whisper"`,
    `"deepgram"`). When given, transcription is delegated to scribed
    (which resolves that engine’s own credentials); the LiteLLM path is
    bypassed. Requires `pip install 'aix[scribed]'`. See
    `scribed.list_backends()`.
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Transcription model (e.g. `'whisper-1'`); for a scribed engine,
    the engine-specific model (e.g. a Whisper size).
  * **language** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Source language (ISO-639-1 code, e.g. `'en'`, `'es'`).
  * **prompt** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Optional text to guide the model’s style (LiteLLM path).
  * **response_format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – `'text'` (default) → `str`; `'srt'`/`'vtt'` →
    subtitle `str` (scribed path); else → [`TranscriptionResult`](#aix.audio.TranscriptionResult).
  * **temperature** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Sampling temperature (LiteLLM path).
  * **timestamp_granularities** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Timestamp types (‘word’, ‘segment’) (LiteLLM path).
  * **\*\*kwargs** – Additional parameters (forwarded to LiteLLM, or to the scribed
    backend — e.g. `diarize=True`).
* **Return type:**
  `Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`TranscriptionResult`](#aix.audio.TranscriptionResult)]
* **Returns:**
  `str` for `response_format` in {text, srt, vtt}, else a
  [`TranscriptionResult`](#aix.audio.TranscriptionResult).

### Examples

```pycon
>>> from aix.audio import transcribe
>>> text = transcribe("recording.mp3")
>>> # delegate to a scribed engine (local, free, diarized SRT):
>>> srt = transcribe(
...     "meeting.wav", engine="faster-whisper", response_format="srt"
... )
>>> dg = transcribe(
...     "call.mp3", engine="deepgram", diarize=True,
...     response_format="verbose_json",
... )
```

### aix.audio.transcribe_with_timestamps(audio, , granularity='segment', model=None, \*\*kwargs)

Transcribe audio with detailed timestamps.

* **Parameters:**
  * **audio** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`BinaryIO`](https://docs.python.org/3/library/typing.html#typing.BinaryIO), [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)]) – Audio file path, file object, or bytes
  * **granularity** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Timestamp granularity (‘word’ or ‘segment’)
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Transcription model
  * **\*\*kwargs** – Additional parameters for transcribe()
* **Return type:**
  [`TranscriptionResult`](#aix.audio.TranscriptionResult)
* **Returns:**
  TranscriptionResult with detailed segments

### Examples

```pycon
>>> from aix.audio import transcribe_with_timestamps
>>> result = transcribe_with_timestamps("lecture.mp3")
>>> for segment in result.segments:
...     start = segment['start']
...     end = segment['end']
...     text = segment['text']
...     print(f"[{start:.2f}-{end:.2f}] {text}")
```

### aix.audio.translate_audio(audio, , model=None, prompt=None, api_key=None, \*\*kwargs)

Translate audio from any language to English.

#### NOTE
Currently uses Whisper’s translation capability which translates to English.

* **Parameters:**
  * **audio** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`BinaryIO`](https://docs.python.org/3/library/typing.html#typing.BinaryIO), [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)]) – Audio file path, file object, or bytes
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Translation model (typically ‘whisper-1’)
  * **prompt** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Optional text to guide translation
  * **\*\*kwargs** – Additional provider-specific parameters
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  Translated text in English

### Examples

```pycon
>>> from aix.audio import translate_audio
>>> english_text = translate_audio("spanish_audio.mp3")
>>> print(english_text)
'This is the English translation.'
```
