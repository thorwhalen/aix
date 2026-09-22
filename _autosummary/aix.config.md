# aix.config

Central configuration for AIX (single source of truth for defaults).

This module is the SSOT for \*what model (and related parameters) each AIX
function uses by default\*. Instead of scattering hardcoded `DFLT_*` constants
across modules, every function resolves its defaults from a single, layered
configuration.

Resolution layers, highest precedence first:

1. **Explicit call argument** – e.g. `chat(..., model=...)` always wins
   (handled at the call site, not here).
2. **Runtime override** – [`configure()`](#aix.config.configure) swaps the active config; [`using()`](#aix.config.using)
   scopes an override to a `with` block.
3. **Environment variables** – `AIX_CHAT_MODEL`, `AIX_CHAT_TEMPERATURE`,
   `AIX_EMBEDDING_MODEL`, `AIX_IMAGE_MODEL`, `AIX_TTS_MODEL`,
   `AIX_TTS_VOICE`, `AIX_TRANSCRIPTION_MODEL`, …
4. **User config file** – a TOML file at `<app config dir>/config.toml`
   (override the path with the `AIX_CONFIG_FILE` environment variable).
5. **Shipped defaults** – the dataclass field defaults below.

The schema is a set of frozen dataclasses (immutable). A module-level \*active
config\* pointer holds the currently resolved [`AixConfig`](#aix.config.AixConfig); [`configure()`](#aix.config.configure)
and [`using()`](#aix.config.using) swap that pointer rather than mutating in place.

### Examples

```pycon
>>> from aix import config
>>> c = config.get_config()
>>> isinstance(c.chat.model, str)
True
```

Scoped override (does not leak outside the block):

```pycon
>>> with config.using(chat_model="some/model"):
...     config.get_config().chat.model
'some/model'
>>> config.get_config().chat.model == c.chat.model
True
```

TOML config file shape:

```default
[chat]
model = "openai/gpt-4o-mini"
temperature = 1.0

[embeddings]
model = "text-embedding-3-small"

[image]
model = "dall-e-3"
size = "1024x1024"

[audio]
tts_model = "tts-1"
tts_voice = "alloy"
transcription_model = "whisper-1"

[vision]
model = "gpt-4o-mini"

[aliases]
fast = "openai/gpt-4o-mini"
best = "anthropic/claude-sonnet-4"
```

### Functions

| [`load_config`](#aix.config.load_config)([path, environ])       | Resolve an [`AixConfig`](#aix.config.AixConfig) from shipped defaults, TOML file, and env.   |
|-------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| [`get_config`](#aix.config.get_config)()                       | Return the current active [`AixConfig`](#aix.config.AixConfig).                              |
| [`set_config`](#aix.config.set_config)(config)                 | Replace the active config wholesale.                                                                               |
| [`configure`](#aix.config.configure)(\*\*overrides)           | Apply runtime overrides to the active config and return it.                                                        |
| [`using`](#aix.config.using)(\*\*overrides)               | Context manager applying scoped overrides, restored on exit.                                                       |
| [`config_file_path`](#aix.config.config_file_path)()                 | Path to the user config file (env `AIX_CONFIG_FILE` overrides default).                                            |
| [`resolve_model`](#aix.config.resolve_model)(model, \*[, config]) | Resolve a semantic alias (`"fast"`, `"best"`, ...) to a concrete model id.                                         |

### Classes

| [`ChatConfig`](#aix.config.ChatConfig)([model, temperature, max_tokens])     | Defaults for `chat` / `ask` / `prompt_func`.                       |
|---------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| [`EmbeddingConfig`](#aix.config.EmbeddingConfig)([model, batch_size])             | Defaults for `embeddings` / `embed`.                               |
| [`ImageConfig`](#aix.config.ImageConfig)([model, size, quality, num_images])  | Defaults for `generate_image` / `generate_images`.                 |
| [`AudioConfig`](#aix.config.AudioConfig)([tts_model, tts_voice, ...])         | Defaults for `text_to_speech` / `transcribe` / `translate_audio`.  |
| [`VideoConfig`](#aix.config.VideoConfig)([provider])                          | Defaults for `generate_video` (provider-dependent).                |
| [`VisionConfig`](#aix.config.VisionConfig)([model])                            | Defaults for `describe_image` (image→text).                        |
| [`AixConfig`](#aix.config.AixConfig)([chat, embeddings, image, audio, ...]) | Top-level AIX configuration (single source of truth for defaults). |

### *class* aix.config.AixConfig(chat=<factory>, embeddings=<factory>, image=<factory>, audio=<factory>, video=<factory>, vision=<factory>, aliases=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Top-level AIX configuration (single source of truth for defaults).

### *class* aix.config.AudioConfig(tts_model='gpt-4o-mini-tts', tts_voice='alloy', tts_speed=1.0, transcription_model='whisper-1')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Defaults for `text_to_speech` / `transcribe` / `translate_audio`.

### *class* aix.config.ChatConfig(model='gpt-4.1-mini', temperature=1.0, max_tokens=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Defaults for `chat` / `ask` / `prompt_func`.

### *class* aix.config.EmbeddingConfig(model='text-embedding-3-small', batch_size=512)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Defaults for `embeddings` / `embed`.

### *class* aix.config.ImageConfig(model='dall-e-3', size='1024x1024', quality='standard', num_images=1)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Defaults for `generate_image` / `generate_images`.

### *class* aix.config.VideoConfig(provider=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Defaults for `generate_video` (provider-dependent).

### *class* aix.config.VisionConfig(model='gpt-4o-mini')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Defaults for `describe_image` (image→text).

The default is a cheap, vision-capable OpenAI model so it works with the
same `OPENAI_API_KEY` most users already have; override per call
(`describe_image(..., model=...)`), via `configure(vision_model=...)`,
or the `[vision]` TOML section to route to Claude / Gemini / etc.

### aix.config.config_file_path()

Path to the user config file (env `AIX_CONFIG_FILE` overrides default).

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### aix.config.configure(\*\*overrides)

Apply runtime overrides to the active config and return it.

* **Return type:**
  [`AixConfig`](#aix.config.AixConfig)

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

### aix.config.get_config()

Return the current active [`AixConfig`](#aix.config.AixConfig).

* **Return type:**
  [`AixConfig`](#aix.config.AixConfig)

### aix.config.load_config(path=None, , environ=None)

Resolve an [`AixConfig`](#aix.config.AixConfig) from shipped defaults, TOML file, and env.

Precedence (low to high): shipped defaults < TOML file < environment variables.
Runtime overrides ([`configure()`](#aix.config.configure)/[`using()`](#aix.config.using)) and explicit call arguments
sit above this and are applied elsewhere.

* **Parameters:**
  * **path** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Optional explicit TOML path. Defaults to [`config_file_path()`](#aix.config.config_file_path).
  * **environ** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Mapping`](https://docs.python.org/3/library/typing.html#typing.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Optional environment mapping (defaults to `os.environ`).
* **Return type:**
  [`AixConfig`](#aix.config.AixConfig)
* **Returns:**
  A fully resolved [`AixConfig`](#aix.config.AixConfig).

### aix.config.resolve_model(model, , config=None)

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

### aix.config.set_config(config)

Replace the active config wholesale. Returns the new active config.

* **Return type:**
  [`AixConfig`](#aix.config.AixConfig)

### aix.config.using(\*\*overrides)

Context manager applying scoped overrides, restored on exit.

* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/typing.html#typing.Iterator)[[`AixConfig`](#aix.config.AixConfig)]

### Examples

```pycon
>>> from aix import config
>>> with config.using(chat_temperature=0.0) as c:
...     c.chat.temperature
0.0
```
