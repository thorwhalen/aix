# aix

AIX: Artificial Intelligence eXtensions

A clean, pythonic facade for common AI operations that abstracts away
provider-specific details and complexities.

Quick Start:

> ```pycon
> >>> from aix import chat, embeddings, prompt_func, models
> ```

> ### Simple chat

> ```pycon
> >>> response = chat("What is 2+2?")
> 'The answer is 4.'
> ```

> ### Create prompt-based functions

> ```pycon
> >>> translate = prompt_func("Translate to French: {text}")
> >>> translate(text="Hello world")
> 'Bonjour le monde'
> ```

> ### Get embeddings

> ```pycon
> >>> vecs = list(embeddings(["hello", "world"]))
> >>> len(vecs)
> 2
> ```

> ### Discover models

> ```pycon
> >>> models.discover()
> >>> list(models)[:5]
> ['openai/gpt-4o', 'openai/gpt-4o-mini', ...]
> ```

Main Features:

> - chat(): Simple chat interface across providers
> - embeddings(): Vector embeddings for text
> - prompt_func(): Create functions from prompt templates
> - models: Model discovery and selection
> - generate_image(): Text-to-image generation
> - text_to_speech(), transcribe(): Audio operations
> - generate_video(): Text-to-video generation (provider-dependent)
> - Batch operations for efficiency
> - Clean, i2mint-style Mapping interfaces

Backends:

> - Uses LiteLLM for provider interactions
> - Supports OpenAI, Anthropic, Google, and 100+ models
> - OpenRouter integration for multi-provider access

For detailed documentation, see: [https://github.com/thorwhalen/aix](https://github.com/thorwhalen/aix)

### Functions

| [`get_config`](#aix.get_config)()                                       | Return the current active [`AixConfig`](#aix.AixConfig).                            |
|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| [`set_config`](#aix.set_config)(config)                                 | Replace the active config wholesale.                                                                             |
| [`configure`](#aix.configure)(\*\*overrides)                           | Apply runtime overrides to the active config and return it.                                                      |
| [`using`](#aix.using)(\*\*overrides)                               | Context manager applying scoped overrides, restored on exit.                                                     |
| [`load_config`](#aix.load_config)([path, environ])                       | Resolve an [`AixConfig`](#aix.AixConfig) from shipped defaults, TOML file, and env. |
| [`resolve_model`](#aix.resolve_model)(model, \*[, config])                 | Resolve a semantic alias (`"fast"`, `"best"`, ...) to a concrete model id.                                       |
| [`resolve_api_key`](#aix.resolve_api_key)(model_or_provider, \*[, ...])      | Resolve an API key for `model_or_provider` through the documented layers.                                        |
| [`check_requirements`](#aix.check_requirements)(model_or_provider, \*[, ...])   | Presence-only preflight: ensure a key for `model_or_provider` is resolvable.                                     |
| [`check_keys`](#aix.check_keys)([providers])                            | Report, per provider, whether a usable API key is discoverable.                                                  |
| [`chat`](#aix.chat)(prompt, \*[, model, temperature, ...])        | Send a chat prompt and get a response.                                                                           |
| [`ask`](#aix.ask)(question[, model])                             | Ask a single question and get an answer.                                                                         |
| [`chat_with_history`](#aix.chat_with_history)([system_prompt, model])          | Create a stateful chat session that maintains conversation history.                                              |
| [`embeddings`](#aix.embeddings)(segments, \*[, model, api_key])         | Generate embeddings for multiple text segments.                                                                  |
| [`embed`](#aix.embed)(text, \*[, model])                           | Generate embedding for a single text.                                                                            |
| [`cosine_similarity`](#aix.cosine_similarity)(vec1, vec2)                      | Compute cosine similarity between two vectors.                                                                   |
| [`find_most_similar`](#aix.find_most_similar)(query, candidates, \*[, ...])    | Find most similar texts to a query.                                                                              |
| [`prompt_func`](#aix.prompt_func)(template, \*[, output_schema, ...])    | Create a callable function from a prompt template.                                                               |
| [`prompt_to_text`](#aix.prompt_to_text)(template, \*\*kwargs)               | Create a function that returns text output.                                                                      |
| [`prompt_to_json`](#aix.prompt_to_json)(template, schema, \*\*kwargs)       | Create a function that returns structured JSON output.                                                           |
| [`constrained_answer`](#aix.constrained_answer)(prompt, valid_answers, \*)      | Get an answer from the LLM constrained to a set of valid answers or types.                                       |
| [`discover_available_models`](#aix.discover_available_models)([source, verbose])       | Discover available models from a source.                                                                         |
| [`get_model_info`](#aix.get_model_info)(model_id)                           | Get information about a specific model.                                                                          |
| [`find_models`](#aix.find_models)(query)                                 | Search for models matching a query.                                                                              |
| [`batch_chat`](#aix.batch_chat)(prompts, \*[, model, batch_size, ...])  | Process multiple chat prompts in batches.                                                                        |
| [`batch_embeddings`](#aix.batch_embeddings)(segments, \*[, model, ...])       | Generate embeddings for multiple texts in batches.                                                               |
| [`batch_process`](#aix.batch_process)(items, process_func, \*[, ...])      | Generic batch processing with parallel execution and retries.                                                    |
| [`generate_image`](#aix.generate_image)(prompt, \*[, model, size, ...])     | Generate a single image from a text prompt.                                                                      |
| [`generate_images`](#aix.generate_images)(prompt, \*[, n, model, size, ...]) | Generate multiple images from a text prompt.                                                                     |
| [`edit_image`](#aix.edit_image)(image_path, prompt, \*[, ...])          | Edit an existing image based on a prompt.                                                                        |
| [`create_variation`](#aix.create_variation)(image_path, \*[, model, ...])     | Create variations of an existing image.                                                                          |
| [`text_to_speech`](#aix.text_to_speech)(text, \*[, model, voice, ...])      | Convert text to speech audio.                                                                                    |
| [`transcribe`](#aix.transcribe)(audio, \*[, engine, model, ...])        | Transcribe audio to text.                                                                                        |
| [`transcribe_with_timestamps`](#aix.transcribe_with_timestamps)(audio, \*[, ...])       | Transcribe audio with detailed timestamps.                                                                       |
| [`translate_audio`](#aix.translate_audio)(audio, \*[, model, prompt, ...])   | Translate audio from any language to English.                                                                    |
| [`generate_video`](#aix.generate_video)(prompt, \*[, model, duration, ...]) | Generate a video from a text prompt.                                                                             |
| [`animate_image_to_video`](#aix.animate_image_to_video)(image_path[, prompt, ...])  | Animate a static image into a video.                                                                             |
| [`extend_video`](#aix.extend_video)(video_path[, prompt, ...])            | Extend an existing video with additional generated content.                                                      |
| [`get_video_providers`](#aix.get_video_providers)()                              | Get list of available video generation providers.                                                                |
| [`describe_image`](#aix.describe_image)(image, \*[, prompt, model, ...])    | Describe (or answer a question about) `image` and return the text.                                               |
| [`to_image_content`](#aix.to_image_content)(image, \*[, detail])              | Build a multimodal `image_url` content block for `image`.                                                        |
| [`compare_images`](#aix.compare_images)(candidate, reference, \*[, ...])    | Compare a `candidate` image to `reference` image(s) on a rubric.                                                 |

### Classes

| [`AixConfig`](#aix.AixConfig)([chat, embeddings, image, audio, ...])   | Top-level AIX configuration (single source of truth for defaults).   |
|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| [`ChatSession`](#aix.ChatSession)([system_prompt, model])                | Stateful chat session that maintains conversation history.           |
| [`EmbeddingCache`](#aix.EmbeddingCache)([model])                            | Cache for embeddings to avoid redundant API calls.                   |
| [`PromptFuncs`](#aix.PromptFuncs)([model])                               | Collection of prompt-based functions.                                |
| [`ModelStore`](#aix.ModelStore)([storage_path, auto_discover])          | User-friendly interface for model discovery and selection.           |
| [`BatchProcessor`](#aix.BatchProcessor)(\*[, batch_size, max_workers, ...]) | Stateful batch processor for managing long-running operations.       |
| [`BatchError`](#aix.BatchError)(exception, \*[, index, message])        | A batch result slot standing in for an item that raised.             |
| [`GeneratedImage`](#aix.GeneratedImage)([url, b64_json, model, ...])        | Wrapper for generated images.                                        |
| [`GeneratedAudio`](#aix.GeneratedAudio)(data[, model, text, voice, ...])    | Wrapper for generated audio.                                         |
| [`TranscriptionResult`](#aix.TranscriptionResult)(text[, language, ...])         | Result of audio transcription.                                       |
| [`GeneratedVideo`](#aix.GeneratedVideo)([url, data, model, prompt, ...])    | Wrapper for generated videos.                                        |
| [`RubricVerdict`](#aix.RubricVerdict)(aspect, match, confidence[, note])   | A vision model's verdict on a single rubric aspect.                  |
| [`ImageComparison`](#aix.ImageComparison)(match, confidence, explanation)    | Structured result of comparing a candidate image to reference(s).    |

### Exceptions

| [`MissingCredentialError`](#aix.MissingCredentialError)(model_or_provider, \*)   | Raised when a required API key cannot be resolved.                   |
|--------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| [`ConstraintViolation`](#aix.ConstraintViolation)(message, \*[, answer, ...]) | Raised when an LLM answer does not satisfy the requested constraint. |

### *class* aix.AixConfig(chat=<factory>, embeddings=<factory>, image=<factory>, audio=<factory>, video=<factory>, vision=<factory>, aliases=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Top-level AIX configuration (single source of truth for defaults).

### *class* aix.BatchError(exception, , index=None, message=None)

Bases: [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

A batch result slot standing in for an item that raised.

Subclasses `str` so it *is* the `"ERROR: ..."` string these functions
have always put in the failed slot – comparisons, `.startswith`,
slicing, JSON serialisation and logging all behave identically. What it
adds is the ability to tell a failure apart from a model that legitimately
replied with text starting “ERROR:”, and to reach the original exception.

#### exception

The exception that was caught.

#### index

Position of the failed item in the input, when known.

### Examples

```pycon
>>> err = BatchError(RuntimeError("rate limit exceeded"), index=1)
>>> err == "ERROR: rate limit exceeded"
True
>>> isinstance(err, str), isinstance(err.exception, RuntimeError)
(True, True)
>>> err.index
1
```

### *class* aix.BatchProcessor(, batch_size=None, max_workers=None, show_progress=True)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Stateful batch processor for managing long-running operations.

Provides a higher-level interface for batch processing with
progress tracking, error handling, and result caching.

### Examples

```pycon
>>> processor = BatchProcessor(show_progress=True)
>>> results = processor.process_chats(prompts)
>>> processor.save_results("output.json")
```

#### clear()

Clear stored results and errors.

#### process_chats(prompts, \*\*kwargs)

Process chat prompts and store results.

* **Parameters:**
  * **prompts** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]]) – Prompts to process
  * **\*\*kwargs** – Additional parameters for batch_chat()
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]
* **Returns:**
  List of results

#### process_embeddings(texts, \*\*kwargs)

Process embeddings and store results.

* **Parameters:**
  * **texts** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Texts to embed
  * **\*\*kwargs** – Additional parameters for batch_embeddings()
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]
* **Returns:**
  List of embedding vectors

#### save_results(filepath)

Save results to file.

* **Parameters:**
  **filepath** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Path to save results (JSON)

### *class* aix.ChatSession(system_prompt=None, , model=None, \*\*chat_kwargs)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Stateful chat session that maintains conversation history.

This class provides a convenient way to have multi-turn conversations
without manually managing message history.

### Examples

```pycon
>>> session = ChatSession()
>>> response = session.send("My name is Alice")
>>> response = session.send("What's my name?")
'Your name is Alice.'
```

#### clear_history(keep_system=True)

Clear conversation history.

* **Parameters:**
  **keep_system** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, preserve system message (if any)

#### send(message, \*\*kwargs)

Send a message and get a response.

* **Parameters:**
  * **message** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – User message to send
  * **\*\*kwargs** – Override chat parameters for this message
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  Assistant’s response

### *exception* aix.ConstraintViolation(message, , answer=None, valid_answers=None)

Bases: [`ValueError`](https://docs.python.org/3/builtins/exceptions.html#ValueError)

Raised when an LLM answer does not satisfy the requested constraint.

A subclass of `ValueError` so that callers who already guard
[`constrained_answer()`](#aix.constrained_answer) with `except ValueError` keep working.

#### answer

The offending value, as it came back from the model.

#### valid_answers

The constraint the value failed, verbatim as passed in.

### Examples

```pycon
>>> violation = ConstraintViolation("nope", answer="nope", valid_answers=["a"])
>>> violation.answer, violation.valid_answers
('nope', ['a'])
>>> isinstance(violation, ValueError)
True
```

### *class* aix.EmbeddingCache(model=None, \*\*embedding_kwargs)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Cache for embeddings to avoid redundant API calls.

Useful when you need to embed the same texts multiple times.

### Examples

```pycon
>>> cache = EmbeddingCache()
>>> vec1 = cache.embed("hello")  # API call
>>> vec2 = cache.embed("hello")  # From cache
>>> vec1 == vec2
True
```

#### clear()

Clear the cache.

#### embed(text, force_refresh=False)

Get embedding for text, using cache if available.

* **Parameters:**
  * **text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Text to embed
  * **force_refresh** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, bypass cache and get fresh embedding
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]
* **Returns:**
  Vector embedding

#### embed_batch(texts, force_refresh=False)

Get embeddings for multiple texts, using cache when possible.

* **Parameters:**
  * **texts** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Texts to embed
  * **force_refresh** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, bypass cache
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]
* **Returns:**
  List of embeddings in same order as input texts

### *class* aix.GeneratedAudio(data, model=None, text=None, voice=None, format='mp3')

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

### *class* aix.GeneratedImage(url=None, b64_json=None, model=None, prompt=None, revised_prompt=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Wrapper for generated images.

Provides convenient access to image data in various formats.

### Examples

```pycon
>>> img = GeneratedImage(url="https://...", model="dall-e-3")
>>> img.save("output.png")
>>> img.show()
>>> data = img.as_bytes()
```

#### as_bytes()

Get image as bytes.

* **Return type:**
  [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)
* **Returns:**
  Image data as bytes

#### as_pil_image()

Get image as PIL Image object.

* **Returns:**
  PIL.Image object
* **Raises:**
  [**ImportError**](https://docs.python.org/3/builtins/exceptions.html#ImportError) – If PIL is not installed

#### save(path, format=None)

Save image to file.

* **Parameters:**
  * **path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Output file path
  * **format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Image format (e.g., ‘PNG’, ‘JPEG’). Auto-detected from path if None.

### Examples

```pycon
>>> img.save("output.png")
>>> img.save("output.jpg", format="JPEG")
```

#### show()

Display the image.

Requires PIL (Pillow) to be installed.

### Examples

```pycon
>>> img.show()
```

### *class* aix.GeneratedVideo(url=None, data=None, model=None, prompt=None, duration=None, resolution=None, status='completed', task_id=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Wrapper for generated videos.

Provides convenient access to video data and metadata.

### Examples

```pycon
>>> video = GeneratedVideo(url="https://...")
>>> video.save("output.mp4")
>>> print(video.duration)
5.0
```

#### as_bytes()

Get video as bytes.

* **Return type:**
  [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)
* **Returns:**
  Video data as bytes

#### save(path)

Save video to file.

* **Parameters:**
  **path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Output file path

### Examples

```pycon
>>> video.save("output.mp4")
```

#### wait_until_complete(max_wait=300, poll_interval=5)

Wait for video generation to complete (for async operations).

* **Parameters:**
  * **max_wait** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum time to wait in seconds
  * **poll_interval** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Time between status checks in seconds
* **Raises:**
  * [**TimeoutError**](https://docs.python.org/3/builtins/exceptions.html#TimeoutError) – If generation doesn’t complete within max_wait
  * [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) – If generation fails

### *class* aix.ImageComparison(match, confidence, explanation, aspects=<factory>, model=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Structured result of comparing a candidate image to reference(s).

Behaves like an ordered, read-only mapping of `aspect -> RubricVerdict`
(`comparison["identity"]`, `in`, iteration, `len`) so per-aspect
lookups read naturally, while also carrying the overall verdict.

#### match

Overall pass/fail across the whole rubric.

#### confidence

Overall confidence in `[0.0, 1.0]`.

#### explanation

A short overall summary of the comparison.

#### aspects

The per-aspect verdicts, one [`RubricVerdict`](#aix.RubricVerdict) per
rubric item, in rubric order.

#### model

The vision model id that produced the verdict.

#### get(aspect, default=None)

Return the verdict for `aspect`, or `default` if absent.

* **Return type:**
  [`RubricVerdict`](aix.vision.html.md#aix.vision.RubricVerdict) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

### *exception* aix.MissingCredentialError(model_or_provider, , provider=None, env_names=None)

Bases: [`Exception`](https://docs.python.org/3/builtins/exceptions.html#Exception)

Raised when a required API key cannot be resolved.

The message names which key is missing, how to set it, and (when known)
where to obtain one. Key *values* are never included.

### *class* aix.ModelStore(storage_path=None, auto_discover=False)

Bases: [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)

User-friendly interface for model discovery and selection.

Provides a Mapping interface over the ModelManager with convenient
access patterns and integration with chat/embeddings functions.

### Examples

```pycon
>>> models = ModelStore()
>>> models.discover()  # Fetch available models
```

```pycon
>>> # List all models
>>> list(models)[:5]
['openai/gpt-4o', 'openai/gpt-4o-mini', ...]
```

```pycon
>>> # Get model info
>>> info = models['openai/gpt-4o']
>>> info.provider
'openai'
```

```pycon
>>> # Filter models
>>> openai_models = models.filter(provider='openai')
>>> local_models = models.filter(is_local=True)
```

```pycon
>>> # Use with chat
>>> from aix.chat import chat
>>> model = models['gpt-4o-mini']
>>> chat("Hello", model=model.id)
'Hello! How can I help you?'
```

#### by_provider(provider)

Get all models from a specific provider.

* **Parameters:**
  **provider** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Provider name
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Model`](aix.ai_models.base.html.md#aix.ai_models.base.Model)]
* **Returns:**
  List of models

### Examples

```pycon
>>> models = ModelStore()
>>> openai_models = models.by_provider('openai')
```

#### by_task(task)

Get models suitable for a specific task.

* **Parameters:**
  **task** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Task name (‘chat’, ‘embedding’, ‘image’, etc.)
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Model`](aix.ai_models.base.html.md#aix.ai_models.base.Model)]
* **Returns:**
  List of suitable models

### Examples

```pycon
>>> models = ModelStore()
>>> chat_models = models.by_task('chat')
```

#### discover(source='openrouter', auto_register=True, verbose=False)

Discover models from a source.

* **Parameters:**
  * **source** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Source name (‘openrouter’, ‘ollama’, etc.)
  * **auto_register** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, add discovered models to registry
  * **verbose** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, print progress information
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Model`](aix.ai_models.base.html.md#aix.ai_models.base.Model)]
* **Returns:**
  List of discovered models

### Examples

```pycon
>>> models = ModelStore()
>>> discovered = models.discover('openrouter')
>>> len(discovered) > 100
True
```

#### filter(, provider=None, is_local=None, min_context_size=None, max_context_size=None, has_capabilities=None, tags=None, custom_filter=None)

Filter models by criteria.

* **Parameters:**
  * **provider** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Filter by provider name (‘openai’, ‘anthropic’, etc.)
  * **is_local** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Filter by local vs remote
  * **min_context_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Minimum context window size
  * **max_context_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum context window size
  * **has_capabilities** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Required capabilities
  * **tags** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Required tags
  * **custom_filter** (`callable`) – Custom filter function: f(Model) -> bool
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Model`](aix.ai_models.base.html.md#aix.ai_models.base.Model)]
* **Returns:**
  List of models matching criteria

### Examples

```pycon
>>> models = ModelStore()
>>> models.discover(verbose=False)
```

```pycon
>>> # Get OpenAI models
>>> openai = models.filter(provider='openai')
```

```pycon
>>> # Get local models
>>> local = models.filter(is_local=True)
```

```pycon
>>> # Get cheap models
>>> cheap = models.filter(
...     custom_filter=lambda m: m.cost_per_token.get('input', 0) < 0.001
... )
```

```pycon
>>> # Combine criteria
>>> good_models = models.filter(
...     provider='openai',
...     min_context_size=8000,
...     custom_filter=lambda m: 'gpt-4' in m.id
... )
```

#### get_connector_metadata(model_id, connector)

Get connector-specific metadata for a model.

* **Parameters:**
  * **model_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Model identifier
  * **connector** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Connector name (‘openai’, ‘openrouter’, etc.)
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]
* **Returns:**
  Dict with connector-specific parameters

### Examples

```pycon
>>> models = ModelStore()
>>> meta = models.get_connector_metadata(
...     'openai/gpt-4o',
...     'openai'
... )
>>> meta['model']
'gpt-4o'
```

#### get_info(model_id)

Get detailed information about a model.

* **Parameters:**
  **model_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Model identifier
* **Return type:**
  [`Model`](aix.ai_models.base.html.md#aix.ai_models.base.Model)
* **Returns:**
  Model object with full metadata

### Examples

```pycon
>>> models = ModelStore()
>>> models.discover(verbose=False)
>>> info = models.get_info('openai/gpt-4o')
>>> info.context_size
128000
```

#### recommend(, task='chat', max_cost_per_mtok=None, min_context_size=None, prefer_local=False)

Get recommended models based on requirements.

* **Parameters:**
  * **task** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Primary task (‘chat’, ‘embedding’, etc.)
  * **max_cost_per_mtok** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Maximum cost per million tokens
  * **min_context_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Minimum required context size
  * **prefer_local** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Prefer local models if available
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Model`](aix.ai_models.base.html.md#aix.ai_models.base.Model)]
* **Returns:**
  List of recommended models, sorted by suitability

### Examples

```pycon
>>> models = ModelStore()
>>> recommendations = models.recommend(
...     task='chat',
...     max_cost_per_mtok=5.0,
...     min_context_size=16000
... )
```

#### search(query)

Search models by text query.

Searches in model ID, provider, and tags.

* **Parameters:**
  **query** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Search query
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Model`](aix.ai_models.base.html.md#aix.ai_models.base.Model)]
* **Returns:**
  List of matching models

### Examples

```pycon
>>> models = ModelStore()
>>> models.discover(verbose=False)
>>> results = models.search('gpt-4')
>>> len(results) > 0
True
```

### *class* aix.PromptFuncs(model=None, \*\*default_kwargs)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Collection of prompt-based functions.

Provides a namespace for organizing related prompt functions with
attribute-based access.

### Examples

```pycon
>>> funcs = PromptFuncs()
>>> funcs.add('summarize', "Summarize: {text}")
>>> funcs.add('translate', "Translate {text} to {language}")
>>> funcs.summarize(text="Long article...")
'Summary...'
>>> funcs.translate(text="Hello", language="Spanish")
'Hola'
```

#### add(name, template, , output_schema=None, \*\*kwargs)

Add a function to the collection.

* **Parameters:**
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Function name (will be accessible as attribute)
  * **template** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Prompt template
  * **output_schema** (`Union`[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict), [`type`](https://docs.python.org/3/builtins/functions.html#type)]) – Optional schema for structured output
  * **\*\*kwargs** – Additional parameters for prompt_func
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### keys()

Get all function names.

### *class* aix.RubricVerdict(aspect, match, confidence, note='')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A vision model’s verdict on a single rubric aspect.

#### aspect

The rubric item this verdict is about (e.g. `"identity"`).

#### match

Whether the candidate matches the reference for this aspect.

#### confidence

The model’s self-reported confidence in `[0.0, 1.0]`.

#### note

A short free-text explanation of the verdict.

### *class* aix.TranscriptionResult(text, language=None, duration=None, segments=None, model=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Result of audio transcription.

Contains the transcribed text and optional metadata like segments and timestamps.

### Examples

```pycon
>>> result = TranscriptionResult(text="Hello world")
>>> print(result.text)
'Hello world'
```

### aix.animate_image_to_video(image_path, prompt=None, , model=None, duration=3.0, motion_strength=0.5, \*\*kwargs)

Animate a static image into a video.

* **Parameters:**
  * **image_path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Path to the source image
  * **prompt** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Optional text prompt to guide the animation
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Video generation model to use
  * **duration** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Animation duration in seconds
  * **motion_strength** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Strength of motion (0.0 to 1.0)
  * **\*\*kwargs** – Additional provider-specific parameters
* **Return type:**
  [`GeneratedVideo`](aix.video.html.md#aix.video.GeneratedVideo)
* **Returns:**
  GeneratedVideo object

### Examples

```pycon
>>> from aix.video import animate_image
>>> video = animate_image(
...     "landscape.jpg",
...     prompt="Gentle camera pan across the scene",
...     duration=4
... )
>>> video.save("animated_landscape.mp4")
```

### aix.ask(question, model=None, \*\*kwargs)

Ask a single question and get an answer.

This is a convenience wrapper around chat() for simple Q&A.

* **Parameters:**
  * **question** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The question to ask
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Model to use
  * **\*\*kwargs** – Additional parameters for chat()
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  The answer as a string

### Examples

```pycon
>>> from aix.chat import ask
>>> ask("What is the capital of France?")
'The capital of France is Paris.'
```

### aix.batch_chat(prompts, , model=None, batch_size=None, max_workers=None, show_progress=False, on_error='return', \*\*chat_kwargs)

Process multiple chat prompts in batches.

This function processes multiple prompts efficiently by:

1. Chunking prompts into batches
2. Processing batches in parallel where possible
3. Yielding results in the same order as input

* **Parameters:**
  * **prompts** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]]) – Iterable of prompts (strings or message lists)
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Model to use for all prompts
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of prompts to process in each batch
  * **max_workers** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum number of parallel workers
  * **show_progress** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, print progress information
  * **on_error** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'return'`, `'raise'`, `'skip'`]) – What to do with a prompt whose call raised.
    `'return'` (default) yields a `BatchError` in that slot, keeping
    the stream the same length and order as the input.
    `'raise'` stops after the batch in which the first failure landed
    and re-raises the lowest-indexed exception of that batch.
    `'skip'` omits the failed slots from the stream.
  * **\*\*chat_kwargs** – Additional parameters passed to chat()
* **Yields:**
  Responses in the same order as input prompts. A prompt that raised
  yields a `BatchError` – a `str` subclass equal to `"ERROR: <message>"`,
  carrying the original exception on `.exception`.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If `on_error` is not one of ‘return’, ‘raise’, ‘skip’.
* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### Examples

```pycon
>>> from aix.batches import batch_chat
>>> prompts = [
...     "What is 2+2?",
...     "What is 3+3?",
...     "What is 5+5?"
... ]
>>> results = list(batch_chat(prompts))
>>> len(results)
3
```

```pycon
>>> # With specific model
>>> results = list(batch_chat(
...     prompts,
...     model="gpt-4o-mini",
...     batch_size=5
... ))
```

```pycon
>>> # Process large dataset
>>> def generate_prompts():
...     for i in range(100):
...         yield f"Explain concept {i}"
>>> results = batch_chat(
...     generate_prompts(),
...     show_progress=True
... )
>>> for i, result in enumerate(results):
...     print(f"Result {i}: {result[:50]}...")
```

```pycon
>>> # Tell a failure apart from a reply that starts with "ERROR:"
>>> for result in batch_chat(prompts):
...     if isinstance(result, BatchError):
...         raise result.exception
```

### aix.batch_embeddings(segments, , model=None, batch_size=None, show_progress=False, \*\*embedding_kwargs)

Generate embeddings for multiple texts in batches.

For efficiency, this function processes embeddings in chunks,
as most embedding APIs can handle multiple texts per request.

* **Parameters:**
  * **segments** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Iterable of text strings to embed
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Embedding model to use
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of texts per batch
  * **show_progress** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, print progress information
  * **\*\*embedding_kwargs** – Additional parameters for embeddings()
* **Yields:**
  Vector embeddings in the same order as input
* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]

### Examples

```pycon
>>> from aix.batches import batch_embeddings
>>> texts = ["hello", "world", "foo", "bar"] * 25  # 100 texts
>>> vectors = list(batch_embeddings(
...     texts,
...     batch_size=10,
...     show_progress=True
... ))
>>> len(vectors)
100
```

```pycon
>>> # Process large dataset efficiently
>>> def read_documents():
...     # Generator that yields documents
...     for i in range(1000):
...         yield f"Document {i} content"
>>> all_vectors = []
>>> for vec in batch_embeddings(read_documents()):
...     all_vectors.append(vec)
```

### aix.batch_process(items, process_func, , batch_size=None, max_workers=None, show_progress=False, retry_attempts=None, retry_delay=None, on_error='return')

Generic batch processing with parallel execution and retries.

This is a general-purpose batch processor that can be used for
any operation, not just chat or embeddings.

* **Parameters:**
  * **items** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – Items to process
  * **process_func** (`callable`) – Function to apply to each item
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Batch size for chunking
  * **max_workers** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum parallel workers
  * **show_progress** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Show progress information
  * **retry_attempts** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of retry attempts on failure
  * **retry_delay** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Delay between retries (seconds)
  * **on_error** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'return'`, `'raise'`, `'skip'`]) – What to do with an item still failing after every retry.
    `'return'` (default) yields a `BatchError` in that slot, `'raise'`
    re-raises, `'skip'` omits it. See [`batch_chat()`](#aix.batch_chat).
* **Yields:**
  Results in same order as input. An item that raised yields a
  `BatchError` – a `str` subclass equal to `"ERROR: <message>"`,
  carrying the original exception on `.exception`.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If `on_error` is not one of ‘return’, ‘raise’, ‘skip’.
* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### Examples

```pycon
>>> from aix.batches import batch_process
>>> from aix.chat import chat
```

```pycon
>>> # Custom processing function
>>> def analyze_sentiment(text):
...     return chat(f"Analyze sentiment: {text}")
```

```pycon
>>> texts = ["I love this!", "This is terrible", "It's okay"]
>>> results = list(batch_process(
...     texts,
...     analyze_sentiment,
...     batch_size=5
... ))
```

```pycon
>>> # With retries for flaky operations
>>> def flaky_api_call(item):
...     # Some API that might fail
...     return call_api(item)
```

```pycon
>>> results = batch_process(
...     items,
...     flaky_api_call,
...     retry_attempts=3,
...     retry_delay=2.0
... )
```

### aix.chat(prompt, , model=None, temperature=None, max_tokens=None, stream=False, api_key=None, \*\*kwargs)

Send a chat prompt and get a response.

This is the main chat interface for AIX. It abstracts away provider-specific
details and provides a clean, consistent API across all models.

* **Parameters:**
  * **prompt** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]) – Either a string (becomes a user message) or a list of message dicts
    with ‘role’ and ‘content’ keys
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Model identifier (e.g., ‘gpt-4o’, ‘claude-sonnet-4’,
    ‘openrouter/anthropic/claude-3.5-sonnet’). If None, uses default.
  * **temperature** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Sampling temperature (0.0 = deterministic, 2.0 = creative).
    If None, uses default (1.0).
  * **max_tokens** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum tokens to generate. If None, uses model’s default.
  * **stream** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, return an iterator of text chunks. If False, return
    complete response as string.
  * **api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Explicit API key. If None, resolved from the environment / .env
    / AIX config store for the model’s provider (see aix.credentials).
  * **\*\*kwargs** – Additional provider-specific parameters passed to LiteLLM
* **Returns:**
  Complete response as string
  If stream=True: Iterator yielding text chunks as they arrive
* **Return type:**
  `Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]
* **Raises:**
  * [**ImportError**](https://docs.python.org/3/builtins/exceptions.html#ImportError) – If LiteLLM is not installed
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If prompt format is invalid

### Examples

```pycon
>>> chat("What is Python?")
'Python is a high-level programming language...'
```

```pycon
>>> chat("Hello", model="gpt-4o")
'Hello! How can I assist you today?'
```

```pycon
>>> # Streaming response
>>> for chunk in chat("Count to 5", stream=True):
...     print(chunk, end='', flush=True)
1, 2, 3, 4, 5
```

```pycon
>>> # With message history
>>> messages = [
...     {"role": "system", "content": "You are a helpful assistant."},
...     {"role": "user", "content": "What is 2+2?"}
... ]
>>> chat(messages)
'2+2 equals 4.'
```

### aix.chat_with_history(system_prompt=None, , model=None, \*\*chat_kwargs)

Create a stateful chat session that maintains conversation history.

* **Parameters:**
  * **system_prompt** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Optional system message to set context/behavior
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Model to use for this session
  * **\*\*chat_kwargs** – Additional parameters passed to chat()
* **Return type:**
  [`ChatSession`](#aix.ChatSession)
* **Returns:**
  ChatSession object with send() method

### Examples

```pycon
>>> session = chat_with_history("You are a helpful math tutor")
>>> session.send("What is 2+2?")
'The answer is 4.'
>>> session.send("And if I add 3 to that?")
'That would be 7.'
>>> len(session.history)
5
```

### aix.check_keys(providers=None)

Report, per provider, whether a usable API key is discoverable.

Never returns or logs key *values* – only availability and the env-var
name(s) checked. Useful for quick setup debugging.

* **Parameters:**
  **providers** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Providers to check; defaults to every provider in
  `PROVIDER_ENV_VARS`.
* **Returns:**
  bool, “env_vars”: […],
  “source”: <where-found-or-None>}\`\`. `source` is `"env"`, `"store"`,
  or `None` (never the value itself).
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

### Examples

```pycon
>>> report = check_keys(["openai"])
>>> report["openai"]["available"]
True
```

### aix.check_requirements(model_or_provider, , api_key=None)

Presence-only preflight: ensure a key for `model_or_provider` is resolvable.

Does *not* validate the key over the network – only that one is discoverable
via [`resolve_api_key()`](#aix.resolve_api_key) (explicit arg, env/.env, or store). Raises
[`MissingCredentialError`](#aix.MissingCredentialError) with actionable guidance when absent.

Returns `True` when a key is available.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### Examples

```pycon
>>> check_requirements("gpt-4o", api_key="sk-explicit")
True
```

### aix.compare_images(candidate, reference, , rubric=('identity', 'costume', 'setting', 'lighting', 'props'), model=None, api_key=None, max_tokens=None, temperature=0.0, detail=None, instruction='You are a strict visual continuity supervisor. The FIRST image is the CANDIDATE; the remaining image(s) are the locked REFERENCE the candidate must match. Judge each rubric aspect INDEPENDENTLY (do not let one aspect color another). For each aspect decide whether the candidate matches the reference, give a confidence in [0.0, 1.0], and a short note explaining what matches or drifts. Then give an overall match (true only if every important aspect matches), an overall confidence, and a one-sentence explanation.', \*\*kwargs)

Compare a `candidate` image to `reference` image(s) on a rubric.

The explainable half of a reference supervisor: a vision model returns a
per-aspect pass/fail checklist (does the face match? the costume? the set?
the lighting? the props?) plus an overall verdict — the explainable layer
over a cheap numeric identity-cosine gate (which lives elsewhere, e.g.
lookbook). Built on [`to_image_content()`](#aix.to_image_content) (multi-image content block)
and the same multimodal `completion` path as [`describe_image()`](#aix.describe_image),
asked via a JSON contract for a structured answer.

The comparison is *pointwise* (each aspect judged on its own), not a ranked
pairwise comparison, to avoid position bias.

* **Parameters:**
  * **candidate** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – The image under review (URL / path / bytes / PIL image /
    `data:` URI — anything [`to_image_content()`](#aix.to_image_content) accepts).
  * **reference** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any), [`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – The locked reference — a single image *or* a sequence of
    images (a locked set) in the same flexible formats. An empty
    sequence is an error.
  * **rubric** ([`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – The aspects to evaluate, one verdict per item. Defaults to
    `DFLT_COMPARE_RUBRIC`; pass `DFLT_FILM_RUBRIC` (or any
    custom sequence) to override — e.g. `("face", "architecture",
    "props", "lighting")`. Must be non-empty.
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Vision-capable model id or alias. `None` → the configured
    default ([`aix.config.VisionConfig`](aix.config.html.md#aix.config.VisionConfig)).
  * **api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Explicit API key; `None` resolves it for the model’s
    provider from the environment / `.env` / config store.
  * **max_tokens** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Cap on generated tokens (`None` → provider default).
  * **temperature** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Sampling temperature (default `0.0` for a stable,
    reproducible verdict; `None` → provider default).
  * **detail** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Image detail hint (`"low"` | `"high"` | `"auto"`),
    applied to every image block.
  * **instruction** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The system-style framing prepended to the rubric. Has a
    sensible default; override to retune the supervisor’s strictness.
  * **\*\*kwargs** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Extra provider-specific params forwarded to LiteLLM.
* **Return type:**
  [`ImageComparison`](aix.vision.html.md#aix.vision.ImageComparison)
* **Returns:**
  An [`ImageComparison`](#aix.ImageComparison) — overall `match` / `confidence` /
  `explanation` plus an ordered, mapping-like collection of
  [`RubricVerdict`](#aix.RubricVerdict) (one per rubric aspect, keyed by aspect).
* **Raises:**
  * [**ImportError**](https://docs.python.org/3/builtins/exceptions.html#ImportError) – If LiteLLM is not installed.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If `rubric` is empty or `reference` is an empty
        sequence, or if the model’s reply can’t be parsed as the expected
        JSON verdict.

### Examples

Default rubric, single reference:

```pycon
>>> compare_images("gen.png", "ref.png")
ImageComparison(match=True, confidence=0.9, ...)
```

Filmmaking rubric, a locked reference *set*:

```pycon
>>> from aix.vision import DFLT_FILM_RUBRIC
>>> compare_images(
...     candidate="frame_042.png",
...     reference=["ref_front.png", "ref_side.png"],
...     rubric=DFLT_FILM_RUBRIC,
... )["face_identity"].match
True
```

### aix.configure(\*\*overrides)

Apply runtime overrides to the active config and return it.

* **Return type:**
  [`AixConfig`](aix.config.html.md#aix.config.AixConfig)

### Examples

```pycon
>>> from aix import config
>>> _ = config.configure(chat_model="openai/gpt-4o-mini")
>>> config.get_config().chat.model
'openai/gpt-4o-mini'
>>> _ = config.configure(chat={"temperature": 0.2})
>>> config.get_config().chat.temperature
0.2
```

### aix.constrained_answer(prompt, valid_answers, , model=None, temperature=None, enhance_prompt=False, n=1, on_violation='raise', max_retries=0)

Get an answer from the LLM constrained to a set of valid answers or types.

Uses JSON mode to ensure the LLM returns a valid response based on constraints.
More flexible than the oa version - works with any model that supports JSON mode
via LiteLLM.

This can be seen as a facade for some common structured output use cases, as well
as a convenient tool to do response statistics and validation (via n>1).

* **Parameters:**
  * **prompt** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The question or prompt to ask the LLM
  * **valid_answers** (`Union`[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`int`](https://docs.python.org/3/builtins/functions.html#int)], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)], [`type`](https://docs.python.org/3/builtins/functions.html#type), [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – 

    Can be:
    - list[str]: List of valid string options
    - list[int]: List of valid integer options
    - list[float]: List of valid float options
    - bool: Constrains answer to True or False
    - int: Any integer
    - float: Any number
    - tuple[float, float]: Numerical range (min, max) inclusive
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The model to use for the LLM (default: uses DFLT_CHAT_MODEL)
  * **temperature** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Temperature for sampling (default: None, uses model’s default).
    Higher values (e.g., 1.0) give more random/varied results.
    Lower values (e.g., 0.0) give more deterministic results.
  * **enhance_prompt** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, adds explicit instructions to the prompt about
    JSON formatting and constraints. If False (default), relies on
    response_format alone. Default is False to match oa behavior.
  * **n** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of times to call the LLM (default: 1)
  * **on_violation** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'raise'`, `'return'`]) – What to do when the model’s answer does not satisfy
    `valid_answers`. `'raise'` (default) raises `ConstraintViolation`.
    `'return'` returns the model’s value unchecked – the behaviour of
    this function before enforcement existed, kept as an escape hatch.
  * **max_retries** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – How many times to re-ask the model after a violation
    before giving up (default: 0, i.e. ask exactly once). Ignored when
    `on_violation='return'`, which never detects a violation.
* **Returns:**
  One of the valid answers, respecting the type constraint.
  If n > 1, returns a list of answers.
* **Raises:**
  * [**ConstraintViolation**](#aix.ConstraintViolation) – If the answer is not a member of an options list,
        falls outside a `(min, max)` range, or cannot be converted to the
        declared type – unless `on_violation='return'`.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If the response is not JSON, or has no “answer” field.

### Examples

```pycon
>>> # String options
>>> answer = constrained_answer(
...     "Is Python compiled or interpreted?",
...     ["compiled", "interpreted", "both"]
... )
>>> answer in ["compiled", "interpreted", "both"]
True
```

```pycon
>>> # Boolean
>>> answer = constrained_answer(
...     "Is Python dynamically typed?",
...     bool
... )
>>> isinstance(answer, bool)
True
```

```pycon
>>> # Integer options
>>> answer = constrained_answer(
...     "How many wheels does a car have?",
...     [2, 3, 4, 6, 8]
... )
>>> answer in [2, 3, 4, 6, 8]
True
```

```pycon
>>> # Numerical range
>>> answer = constrained_answer(
...     "What is a reasonable hourly rate for a senior Python developer? (USD)",
...     (50.0, 300.0)
... )
>>> 50.0 <= answer <= 300.0
True
```

```pycon
>>> # Multiple samples for statistics
>>> answers = constrained_answer(
...     "Which is better: cats or dogs?",
...     ["cats", "dogs"],
...     n=10
... )
>>> len(answers)
10
```

```pycon
>>> # Tolerate an off-constraint answer instead of raising
>>> answer = constrained_answer(
...     "Is Python compiled or interpreted?",
...     ["compiled", "interpreted"],
...     on_violation="return",
... )
```

### aix.cosine_similarity(vec1, vec2)

Compute cosine similarity between two vectors.

* **Parameters:**
  * **vec1** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]) – First vector
  * **vec2** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]) – Second vector
* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)
* **Returns:**
  Cosine similarity (between -1 and 1)

### Examples

```pycon
>>> from aix.embeddings import embed, cosine_similarity
>>> v1 = embed("cat")
>>> v2 = embed("kitten")
>>> similarity = cosine_similarity(v1, v2)
>>> similarity > 0.8
True
```

### aix.create_variation(image_path, , model=None, size=None, n=1, api_key=None, \*\*kwargs)

Create variations of an existing image.

* **Parameters:**
  * **image_path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Path to the source image
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Model to use (typically ‘dall-e-2’)
  * **size** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Output image size
  * **n** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of variations to generate
  * **\*\*kwargs** – Additional provider-specific parameters
* **Return type:**
  `Union`[[`GeneratedImage`](aix.image.html.md#aix.image.GeneratedImage), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`GeneratedImage`](aix.image.html.md#aix.image.GeneratedImage)]]
* **Returns:**
  GeneratedImage or list of GeneratedImage objects

### Examples

```pycon
>>> from aix.image import create_variation
>>> variations = create_variation(
...     "original.png",
...     n=3,
...     size="512x512"
... )
>>> for i, var in enumerate(variations):
...     var.save(f"variation_{i}.png")
```

### aix.describe_image(image, , prompt='Describe this image in detail.', model=None, api_key=None, max_tokens=None, temperature=None, detail=None, \*\*kwargs)

Describe (or answer a question about) `image` and return the text.

The image→text primitive. `image` accepts a URL, a local path, raw bytes,
a PIL image, or a `data:` URI (see [`to_image_content()`](#aix.to_image_content)). `prompt` is
the instruction (default: a generic “describe this image”); pass a question
for VQA or a rubric for a judgement. Everything past `image` is keyword.

* **Parameters:**
  * **image** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – The image (URL / path / bytes / PIL image / `data:` URI).
  * **prompt** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The text instruction accompanying the image.
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Vision-capable model id (e.g. `gpt-4o`, `claude-sonnet-4-6`,
    `gemini/gemini-1.5-pro`) or an alias. `None` → the configured
    default ([`aix.config.VisionConfig`](aix.config.html.md#aix.config.VisionConfig)).
  * **api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Explicit API key; `None` resolves it from the environment /
    `.env` / config store for the model’s provider.
  * **max_tokens** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Cap on generated tokens (`None` → provider default).
  * **temperature** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Sampling temperature (`None` → provider default).
  * **detail** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Image detail hint (`"low"` | `"high"` | `"auto"`).
  * **\*\*kwargs** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Extra provider-specific params forwarded to LiteLLM.
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  The model’s text response.
* **Raises:**
  [**ImportError**](https://docs.python.org/3/builtins/exceptions.html#ImportError) – If LiteLLM is not installed.

```pycon
>>> describe_image("cat.jpg", prompt="Caption it.")
'A cat on a sofa.'
```

### aix.discover_available_models(source='openrouter', verbose=True)

Discover available models from a source.

Convenience function that uses the global models instance.

* **Parameters:**
  * **source** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Source name (‘openrouter’, ‘ollama’, etc.)
  * **verbose** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Print progress information
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Model`](aix.ai_models.base.html.md#aix.ai_models.base.Model)]
* **Returns:**
  List of discovered models

### Examples

```pycon
>>> from aix.models import discover_available_models
>>> models = discover_available_models('openrouter')
>>> len(models) > 100
True
```

### aix.edit_image(image_path, prompt, , mask_path=None, model=None, size=None, n=1, api_key=None, \*\*kwargs)

Edit an existing image based on a prompt.

* **Parameters:**
  * **image_path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Path to the image to edit
  * **prompt** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Description of the desired edit
  * **mask_path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Optional path to mask image (transparent areas will be edited)
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Model to use (typically ‘dall-e-2’ for edits)
  * **size** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Output image size
  * **n** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of variations to generate
  * **\*\*kwargs** – Additional provider-specific parameters
* **Return type:**
  `Union`[[`GeneratedImage`](aix.image.html.md#aix.image.GeneratedImage), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`GeneratedImage`](aix.image.html.md#aix.image.GeneratedImage)]]
* **Returns:**
  GeneratedImage or list of GeneratedImage objects

### Examples

```pycon
>>> from aix.image import edit_image
>>> edited = edit_image(
...     "photo.png",
...     "Add a rainbow in the sky",
...     mask_path="sky_mask.png"
... )
>>> edited.save("edited_photo.png")
```

### aix.embed(text, , model=None, \*\*kwargs)

Generate embedding for a single text.

Convenience function for embedding a single text string.

* **Parameters:**
  * **text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Text string to embed
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Model identifier
  * **\*\*kwargs** – Additional parameters for embeddings()
* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]
* **Returns:**
  Vector embedding as sequence of floats

### Examples

```pycon
>>> from aix.embeddings import embed
>>> vec = embed("Hello, world!")
>>> len(vec)
1536
```

```pycon
>>> # Compare similarity
>>> import numpy as np
>>> v1 = np.array(embed("cat"))
>>> v2 = np.array(embed("kitten"))
>>> v3 = np.array(embed("computer"))
>>> # Cosine similarity
>>> np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
0.92  # High similarity
>>> np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3))
0.23  # Low similarity
```

### aix.embeddings(segments, , model=None, api_key=None, \*\*kwargs)

Generate embeddings for multiple text segments.

This is the main embedding interface for AIX. It abstracts away provider-specific
details and provides a clean, consistent API across all embedding models.

* **Parameters:**
  * **segments** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Iterable of text strings to embed
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Model identifier (e.g., ‘text-embedding-3-small’, ‘text-embedding-ada-002’,
    ‘openrouter/openai/text-embedding-3-small’). If None, uses default.
  * **api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Explicit API key. If None, resolved from the environment / .env
    / AIX config store for the model’s provider (see aix.credentials).
  * **\*\*kwargs** – Additional provider-specific parameters passed to LiteLLM
* **Yields:**
  Vector embeddings as sequences of floats. Each embedding corresponds to
  one input segment in the same order.
* **Raises:**
  * [**ImportError**](https://docs.python.org/3/builtins/exceptions.html#ImportError) – If LiteLLM is not installed
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If segments is empty or invalid
* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]

### Examples

```pycon
>>> from aix.embeddings import embeddings
>>> texts = ["cat", "dog", "bird"]
>>> vecs = list(embeddings(texts))
>>> len(vecs)
3
```

```pycon
>>> # With specific model
>>> vecs = list(embeddings(
...     ["hello", "world"],
...     model="text-embedding-3-large"
... ))
```

```pycon
>>> # Process in chunks for large datasets
>>> def chunk_texts(texts, size=100):
...     for i in range(0, len(texts), size):
...         yield texts[i:i+size]
>>> all_vecs = []
>>> for chunk in chunk_texts(large_dataset):
...     all_vecs.extend(embeddings(chunk))
```

### aix.extend_video(video_path, prompt=None, , extend_duration=2.0, model=None, \*\*kwargs)

Extend an existing video with additional generated content.

* **Parameters:**
  * **video_path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Path to the source video
  * **prompt** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Optional text prompt to guide the extension
  * **extend_duration** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – How many seconds to add
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Video generation model
  * **\*\*kwargs** – Additional provider-specific parameters
* **Return type:**
  [`GeneratedVideo`](aix.video.html.md#aix.video.GeneratedVideo)
* **Returns:**
  GeneratedVideo object with extended content

### Examples

```pycon
>>> from aix.video import extend_video
>>> extended = extend_video(
...     "original.mp4",
...     prompt="Continue the same scene",
...     extend_duration=3
... )
```

### aix.find_models(query)

Search for models matching a query.

Convenience function that uses the global models instance.

* **Parameters:**
  **query** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Search query
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Model`](aix.ai_models.base.html.md#aix.ai_models.base.Model)]
* **Returns:**
  List of matching models

### Examples

```pycon
>>> from aix.models import find_models
>>> results = find_models('claude')
>>> any('claude' in m.id.lower() for m in results)
True
```

### aix.find_most_similar(query, candidates, , model=None, top_k=5, \*\*kwargs)

Find most similar texts to a query.

* **Parameters:**
  * **query** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – Query text or pre-computed embedding vector
  * **candidates** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Candidate texts to compare against
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Embedding model to use
  * **top_k** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of top results to return
  * **\*\*kwargs** – Additional parameters for embeddings()
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]
* **Returns:**
  List of (text, similarity_score) tuples, sorted by similarity (highest first)

### Examples

```pycon
>>> query = "What is machine learning?"
>>> docs = [
...     "Machine learning is a type of AI",
...     "Python is a programming language",
...     "Neural networks are used in deep learning",
... ]
>>> results = find_most_similar(query, docs, top_k=2)
>>> results[0][0]  # Most similar doc
'Machine learning is a type of AI'
```

### aix.generate_image(prompt, , model=None, size=None, quality=None, style=None, response_format='url', api_key=None, \*\*kwargs)

Generate a single image from a text prompt.

* **Parameters:**
  * **prompt** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Text description of the image to generate
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Model to use (e.g., ‘dall-e-2’, ‘dall-e-3’, ‘stable-diffusion’)
  * **size** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Image size (e.g., ‘1024x1024’, ‘512x512’, ‘1792x1024’)
  * **quality** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Image quality (‘standard’ or ‘hd’ for DALL-E 3)
  * **style** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Image style (‘vivid’ or ‘natural’ for DALL-E 3)
  * **response_format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Format of response (‘url’ or ‘b64_json’)
  * **api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Explicit API key. If None, resolved from the environment / .env
    / AIX config store for the model’s provider (see aix.credentials).
  * **\*\*kwargs** – Additional provider-specific parameters
* **Return type:**
  [`GeneratedImage`](aix.image.html.md#aix.image.GeneratedImage)
* **Returns:**
  GeneratedImage object
* **Raises:**
  [**ImportError**](https://docs.python.org/3/builtins/exceptions.html#ImportError) – If LiteLLM is not installed

### Examples

```pycon
>>> from aix.image import generate_image
>>> image = generate_image("A serene mountain landscape")
>>> image.save("landscape.png")
```

```pycon
>>> # High quality with DALL-E 3
>>> image = generate_image(
...     "Abstract art with vibrant colors",
...     model="dall-e-3",
...     quality="hd",
...     style="vivid"
... )
```

```pycon
>>> # Specific size
>>> image = generate_image(
...     "A futuristic city",
...     size="1792x1024"
... )
```

### aix.generate_images(prompt, , n=None, model=None, size=None, quality=None, style=None, response_format='url', api_key=None, \*\*kwargs)

Generate multiple images from a text prompt.

* **Parameters:**
  * **prompt** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Text description of images to generate
  * **n** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of images to generate
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Model to use
  * **size** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Image size
  * **quality** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Image quality
  * **style** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Image style
  * **response_format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Format of response
  * **\*\*kwargs** – Additional provider-specific parameters
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`GeneratedImage`](aix.image.html.md#aix.image.GeneratedImage)]
* **Returns:**
  List of GeneratedImage objects

### Examples

```pycon
>>> from aix.image import generate_images
>>> images = generate_images(
...     "A cute robot",
...     n=3,
...     size="512x512"
... )
>>> for i, img in enumerate(images):
...     img.save(f"robot_{i}.png")
```

### aix.generate_video(prompt, , model=None, duration=5.0, resolution='1280x720', fps=24, aspect_ratio=None, style=None, seed=None, api_key=None, \*\*kwargs)

Generate a video from a text prompt.

#### NOTE
This is a high-level interface. Actual implementation depends on
available video generation providers (Runway, Pika, etc.) and may require
provider-specific API keys.

* **Parameters:**
  * **prompt** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Text description of the video to generate
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Video generation model to use
  * **duration** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Video duration in seconds (typically 2-10 seconds)
  * **resolution** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Video resolution (‘1280x720’, ‘1920x1080’, etc.)
  * **fps** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Frames per second
  * **aspect_ratio** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Aspect ratio (‘16:9’, ‘9:16’, ‘1:1’, etc.)
  * **style** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Video style hint (provider-specific)
  * **seed** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Random seed for reproducibility
  * **api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Explicit API key for the video provider. If None, resolved from
    the environment / .env / AIX config store (see aix.credentials).
  * **\*\*kwargs** – Additional provider-specific parameters
* **Return type:**
  [`GeneratedVideo`](aix.video.html.md#aix.video.GeneratedVideo)
* **Returns:**
  GeneratedVideo object
* **Raises:**
  * [**NotImplementedError**](https://docs.python.org/3/builtins/exceptions.html#NotImplementedError) – If no video provider is configured
  * [**ImportError**](https://docs.python.org/3/builtins/exceptions.html#ImportError) – If required provider SDK is not installed

### Examples

```pycon
>>> from aix.video import generate_video
>>> video = generate_video(
...     "A serene ocean sunset with gentle waves",
...     duration=5,
...     resolution="1920x1080"
... )
>>> video.save("sunset.mp4")
```

```pycon
>>> # Specific style
>>> video = generate_video(
...     "A futuristic city",
...     style="cyberpunk",
...     duration=4
... )
```

### aix.get_config()

Return the current active [`AixConfig`](#aix.AixConfig).

* **Return type:**
  [`AixConfig`](aix.config.html.md#aix.config.AixConfig)

### aix.get_model_info(model_id)

Get information about a specific model.

Convenience function that uses the global models instance.

* **Parameters:**
  **model_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Model identifier
* **Return type:**
  [`Model`](aix.ai_models.base.html.md#aix.ai_models.base.Model)
* **Returns:**
  Model object

### Examples

```pycon
>>> from aix.models import get_model_info
>>> info = get_model_info('openai/gpt-4o')
>>> info.provider
'openai'
```

### aix.get_video_providers()

Get list of available video generation providers.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]
* **Returns:**
  List of provider names that are configured

### Examples

```pycon
>>> from aix.video import get_available_providers
>>> providers = get_available_providers()
>>> print(providers)
['runway', 'pika']
```

### aix.load_config(path=None, , environ=None)

Resolve an [`AixConfig`](#aix.AixConfig) from shipped defaults, TOML file, and env.

Precedence (low to high): shipped defaults < TOML file < environment variables.
Runtime overrides ([`configure()`](#aix.configure)/[`using()`](#aix.using)) and explicit call arguments
sit above this and are applied elsewhere.

* **Parameters:**
  * **path** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Optional explicit TOML path. Defaults to `config_file_path()`.
  * **environ** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Mapping`](https://docs.python.org/3/library/typing.html#typing.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Optional environment mapping (defaults to `os.environ`).
* **Return type:**
  [`AixConfig`](aix.config.html.md#aix.config.AixConfig)
* **Returns:**
  A fully resolved [`AixConfig`](#aix.AixConfig).

### aix.prompt_func(template, , output_schema=None, egress=None, model=None, temperature=None, name=None, \*\*chat_kwargs)

Create a callable function from a prompt template.

This is the main function for creating prompt-based functions. It automatically
detects parameters from the template and creates a function with those parameters.

Without output_schema: Returns text
With output_schema: Returns structured data (dict, list, etc.)
With egress: Returns whatever the egress post-processor returns.

Templates use the default-aware `{name:default}` dialect (matching
`oa.prompt_function`): `{name}` is a required parameter, `{name:default}`
supplies a default value (the text after the colon), and braces inside
`` fenced `` regions are left literal. Plain `{name}`-only templates behave
exactly as before.

```pycon
>>> greet = prompt_func("Greet {name} in {language:English}")
>>> greet.param_names
['name', 'language']
```

* **Parameters:**
  * **template** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Prompt template with {var} placeholders for parameters
  * **output_schema** (`Union`[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict), [`type`](https://docs.python.org/3/builtins/functions.html#type)]) – Optional schema for structured output. Can be:
    - Dict mapping field names to types: {“name”: str, “age”: int}
    - A single type for simple outputs: str, int, list, etc.
    - None for plain text output (default)
  * **egress** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – Optional post-processor `(result) -> Any` applied to the output
    before returning — on both the text and structured paths. Lets a caller
    keep “prompt → typed Python value” inside the facade (e.g. parse the
    LLM text into a list of lines/ids) instead of wrapping the returned
    function. `None` (default) returns the raw result unchanged.
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Model to use for this function
  * **temperature** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Temperature for generation
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Optional `__name__` for the generated function (for tracing /
    identity). Defaults to `"prompt_based_function"`.
  * **\*\*chat_kwargs** – Additional parameters passed to chat()
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)
* **Returns:**
  Callable function with parameters derived from template

### Examples

```pycon
>>> # Simple text generation
>>> summarize = prompt_func("Summarize this text: {text}")
>>> summarize(text="Long article...")
'Brief summary...'
```

```pycon
>>> # Structured output
>>> extract = prompt_func(
...     "Extract contact info from: {text}",
...     output_schema={"name": str, "email": str, "phone": str}
... )
>>> result = extract(text="Contact John at john@example.com, 555-1234")
>>> result['name']
'John'
```

```pycon
>>> # Multiple parameters
>>> compare = prompt_func(
...     "Compare {item1} and {item2} in terms of {aspect}. "
...     "Keep it under {word_limit} words."
... )
>>> compare(
...     item1="Python",
...     item2="Java",
...     aspect="learning curve",
...     word_limit=50
... )
'Python has a gentler learning curve...'
```

```pycon
>>> # With specific model
>>> creative_writer = prompt_func(
...     "Write a creative story about {topic}",
...     model="gpt-4o",
...     temperature=1.5
... )
>>> creative_writer(topic="a time-traveling cat")
'Once upon a time, there was a cat named Whiskers...'
```

### aix.prompt_to_json(template, schema, \*\*kwargs)

Create a function that returns structured JSON output.

This is an explicit alias for prompt_func with output_schema.

* **Parameters:**
  * **template** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Prompt template
  * **schema** (`Union`[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict), [`type`](https://docs.python.org/3/builtins/functions.html#type)]) – Output schema
  * **\*\*kwargs** – Additional parameters for prompt_func
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)
* **Returns:**
  Function that returns structured data

### Examples

```pycon
>>> extract = prompt_to_json(
...     "Extract name and age from: {text}",
...     schema={"name": str, "age": int}
... )
>>> result = extract(text="Alice is 30")
>>> isinstance(result, dict)
True
```

### aix.prompt_to_text(template, \*\*kwargs)

Create a function that returns text output.

This is an explicit alias for prompt_func without output_schema.

* **Parameters:**
  * **template** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Prompt template
  * **\*\*kwargs** – Additional parameters for prompt_func
* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)
* **Returns:**
  Function that returns text

### Examples

```pycon
>>> summarize = prompt_to_text("Summarize: {text}")
>>> result = summarize(text="Long text...")
>>> isinstance(result, str)
True
```

### aix.resolve_api_key(model_or_provider, , api_key=None, prompt_if_missing=False)

Resolve an API key for `model_or_provider` through the documented layers.

Layers, highest precedence first: explicit `api_key` argument, provider
environment variable (with soft `.env` discovery), the AIX config store,
and – only when `prompt_if_missing` is true *and* running interactively –
an interactive prompt that persists the answer.

* **Parameters:**
  * **model_or_provider** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – A model id (provider inferred via LiteLLM) or a
    provider name directly (e.g. `"openai"`).
  * **api_key** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – An explicit key; when given (non-empty) it is returned verbatim.
  * **prompt_if_missing** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If true, fall back to a REPL prompt + persist when no
    key is found elsewhere. Off by default so the common path never
    blocks on input.
* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]
* **Returns:**
  The resolved key, or `None` if genuinely absent.

### Examples

```pycon
>>> resolve_api_key("gpt-4o", api_key="sk-explicit")
'sk-explicit'
```

### aix.resolve_model(model, , config=None)

Resolve a semantic alias (`"fast"`, `"best"`, …) to a concrete model id.

Plain substitution: if `model` is a key in the active `aliases` table it is
replaced (following chains, with cycle protection). Anything that is not an
alias – including every literal model id like `"gpt-4o"` – is returned
unchanged. `None` passes through (callers apply their own default first).

#### NOTE
aliases share a namespace with literal model ids, so an unknown name is
treated as a literal id, not an error. Inspect available aliases via
`aix.get_config().aliases`.

### Examples

```pycon
>>> from aix import config
>>> config.resolve_model("fast") in config.DEFAULT_ALIASES.values()
True
>>> config.resolve_model("gpt-4o")  # not an alias -> unchanged
'gpt-4o'
>>> config.resolve_model(None) is None
True
```

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### aix.set_config(config)

Replace the active config wholesale. Returns the new active config.

* **Return type:**
  [`AixConfig`](aix.config.html.md#aix.config.AixConfig)

### aix.text_to_speech(text, , model=None, voice=None, speed=None, response_format='mp3', api_key=None, \*\*kwargs)

Convert text to speech audio.

* **Parameters:**
  * **text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Text to convert to speech
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – TTS model to use (e.g., ‘tts-1’, ‘tts-1-hd’)
  * **voice** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Voice to use (‘alloy’, ‘echo’, ‘fable’, ‘onyx’, ‘nova’, ‘shimmer’)
  * **speed** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Playback speed (0.25 to 4.0)
  * **response_format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Audio format (‘mp3’, ‘opus’, ‘aac’, ‘flac’)
  * **\*\*kwargs** – Additional provider-specific parameters
* **Return type:**
  [`GeneratedAudio`](aix.audio.html.md#aix.audio.GeneratedAudio)
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

### aix.to_image_content(image, , detail=None)

Build a multimodal `image_url` content block for `image`.

`image` may be:

- an `http(s)://` URL or a `data:` URI — passed through verbatim;
- a local file path (`str` or [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)) — read and inlined
  as a base64 `data:` URI with a guessed MIME type;
- raw `bytes` — base64-inlined (MIME sniffed from magic bytes, else JPEG);
- a PIL `Image` — encoded to PNG and inlined.

`detail` (`"low"` | `"high"` | `"auto"`) is forwarded when set; the
block is the OpenAI/LiteLLM multimodal shape understood across providers.

```pycon
>>> to_image_content("https://x/y.jpg")
{'type': 'image_url', 'image_url': {'url': 'https://x/y.jpg'}}
>>> to_image_content("data:image/png;base64,AAAA")["image_url"]["url"][:10]
'data:image'
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### aix.transcribe(audio, , engine=None, model=None, language=None, prompt=None, response_format='text', temperature=None, timestamp_granularities=None, api_key=None, \*\*kwargs)

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
    subtitle `str` (scribed path); else → [`TranscriptionResult`](#aix.TranscriptionResult).
  * **temperature** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Sampling temperature (LiteLLM path).
  * **timestamp_granularities** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Timestamp types (‘word’, ‘segment’) (LiteLLM path).
  * **\*\*kwargs** – Additional parameters (forwarded to LiteLLM, or to the scribed
    backend — e.g. `diarize=True`).
* **Return type:**
  `Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`TranscriptionResult`](aix.audio.html.md#aix.audio.TranscriptionResult)]
* **Returns:**
  `str` for `response_format` in {text, srt, vtt}, else a
  [`TranscriptionResult`](#aix.TranscriptionResult).

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

### aix.transcribe_with_timestamps(audio, , granularity='segment', model=None, \*\*kwargs)

Transcribe audio with detailed timestamps.

* **Parameters:**
  * **audio** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`BinaryIO`](https://docs.python.org/3/library/typing.html#typing.BinaryIO), [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)]) – Audio file path, file object, or bytes
  * **granularity** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Timestamp granularity (‘word’ or ‘segment’)
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Transcription model
  * **\*\*kwargs** – Additional parameters for transcribe()
* **Return type:**
  [`TranscriptionResult`](aix.audio.html.md#aix.audio.TranscriptionResult)
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

### aix.translate_audio(audio, , model=None, prompt=None, api_key=None, \*\*kwargs)

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

### aix.using(\*\*overrides)

Context manager applying scoped overrides, restored on exit.

* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/typing.html#typing.Iterator)[[`AixConfig`](aix.config.html.md#aix.config.AixConfig)]

### Examples

```pycon
>>> from aix import config
>>> with config.using(chat_temperature=0.0) as c:
...     c.chat.temperature
0.0
```

### Modules

| [`config`](aix.config.html.md#module-aix.config)           | Central configuration for AIX (single source of truth for defaults).   |
|-------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| [`ai_models`](aix.ai_models.html.md#module-aix.ai_models)     | AI Model Management Module.                                            |
| [`audio`](aix.audio.html.md#module-aix.audio)             | Audio operations interface for AIX.                                    |
| [`batches`](aix.batches.html.md#module-aix.batches)         | Batch processing interface for AIX.                                    |
| [`credentials`](aix.credentials.html.md#module-aix.credentials) | Unified, discoverable API-key / secret resolution for AIX.             |
| [`gen_ai`](aix.gen_ai.html.md#module-aix.gen_ai)           | Generative AI modules.                                                 |
| [`image`](aix.image.html.md#module-aix.image)             | Image generation interface for AIX.                                    |
| [`misc`](aix.misc.html.md#module-aix.misc)               | Misc AIX functions.                                                    |
| [`models`](aix.models.html.md#aix.models)                  | User-friendly interface for model discovery and selection.             |
| [`prompts`](aix.prompts.html.md#module-aix.prompts)         | Prompt-based function creation for AIX.                                |
| [`stores`](aix.stores.html.md#module-aix.stores)           | Storage layers for AIX.                                                |
| [`util`](aix.util.html.md#module-aix.util)               | Utils for AIX.                                                         |
| [`video`](aix.video.html.md#module-aix.video)             | Video generation interface for AIX.                                    |
| [`vision`](aix.vision.html.md#module-aix.vision)           | Image-to-text (vision) interface for AIX.                              |
