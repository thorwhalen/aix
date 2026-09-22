# aix.credentials

Unified, discoverable API-key / secret resolution for AIX.

Every key-requiring function in AIX (`chat`, `embeddings`, `generate_image`,
`text_to_speech`, `transcribe`, `generate_video`, …) resolves its
credentials through a single, documented path – instead of relying on LiteLLM’s
implicit environment reading, scattered `os.getenv` calls, or an orphaned
config getter.

Resolution layers, highest precedence first:

1. **Explicit argument** – `chat(..., api_key=...)`.
2. **Provider environment variable** – the canonical name for the inferred
   provider (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`).
   A project `.env` is *softly* discovered when `python-dotenv` is installed
   (no hard dependency); explicit env vars always win over `.env` values.
3. **AIX config store** – a config2py central store in the user app-config
   folder, keyed by the canonical env-var name (so the store and the env layer
   share one namespace). See [`aix.util`](aix.util.html.md#module-aix.util).
4. **(REPL only) interactive prompt** – when `prompt_if_missing=True` *and*
   running interactively, ask once and persist to the config store.

The provider for a model id is inferred via LiteLLM’s `get_llm_provider`; a
small curated `PROVIDER_ENV_VARS` table maps each documented provider to
its env-var name(s) and console URL.

When a required key is genuinely absent, [`check_requirements()`](#aix.credentials.check_requirements) raises
[`MissingCredentialError`](#aix.credentials.MissingCredentialError) – naming *which* key is missing, *how* to set
it, and *where* to obtain one – rather than letting a cryptic, provider-specific
LiteLLM error surface late. This preflight is wired in via the
[`requires_credentials()`](#aix.credentials.requires_credentials) decorator so error-raising stays separate from
business logic.

### Examples

```pycon
>>> from aix.credentials import resolve_api_key, PROVIDER_ENV_VARS
>>> resolve_api_key("gpt-4o", api_key="sk-explicit")  # explicit wins
'sk-explicit'
>>> "openai" in PROVIDER_ENV_VARS
True
```

Security:
: Resolved key *values* are never logged or returned by diagnostics;
  [`check_keys()`](#aix.credentials.check_keys) reports availability only. Keys live in the user
  app-config folder, never in the repo.

### Functions

| [`infer_provider`](#aix.credentials.infer_provider)(model_or_provider)                | Infer the provider name from a model id (or pass through a provider name).   |
|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| [`provider_env_vars`](#aix.credentials.provider_env_vars)(provider)                      | Return the env-var name(s) for `provider` as a list (possibly empty).        |
| [`resolve_api_key`](#aix.credentials.resolve_api_key)(model_or_provider, \*[, ...])    | Resolve an API key for `model_or_provider` through the documented layers.    |
| [`check_requirements`](#aix.credentials.check_requirements)(model_or_provider, \*[, ...]) | Presence-only preflight: ensure a key for `model_or_provider` is resolvable. |
| [`requires_credentials`](#aix.credentials.requires_credentials)(default_model)              | Decorator: preflight credentials before the wrapped function runs.           |
| [`check_keys`](#aix.credentials.check_keys)([providers])                          | Report, per provider, whether a usable API key is discoverable.              |

### Exceptions

| [`MissingCredentialError`](#aix.credentials.MissingCredentialError)(model_or_provider, \*)   | Raised when a required API key cannot be resolved.   |
|--------------------------------------------------------------------------------------------------|------------------------------------------------------|

### *exception* aix.credentials.MissingCredentialError(model_or_provider, , provider=None, env_names=None)

Bases: [`Exception`](https://docs.python.org/3/builtins/exceptions.html#Exception)

Raised when a required API key cannot be resolved.

The message names which key is missing, how to set it, and (when known)
where to obtain one. Key *values* are never included.

### aix.credentials.check_keys(providers=None)

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

### aix.credentials.check_requirements(model_or_provider, , api_key=None)

Presence-only preflight: ensure a key for `model_or_provider` is resolvable.

Does *not* validate the key over the network – only that one is discoverable
via [`resolve_api_key()`](#aix.credentials.resolve_api_key) (explicit arg, env/.env, or store). Raises
[`MissingCredentialError`](#aix.credentials.MissingCredentialError) with actionable guidance when absent.

Returns `True` when a key is available.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### Examples

```pycon
>>> check_requirements("gpt-4o", api_key="sk-explicit")
True
```

### aix.credentials.infer_provider(model_or_provider)

Infer the provider name from a model id (or pass through a provider name).

Uses LiteLLM’s `get_llm_provider` when available. If `model_or_provider`
is already a known provider name (a key in `PROVIDER_ENV_VARS`), it is
returned as-is. Returns `None` if the provider cannot be determined.

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### Examples

```pycon
>>> infer_provider("gpt-4o")
'openai'
>>> infer_provider("openrouter")  # already a provider name
'openrouter'
>>> infer_provider(None) is None
True
```

### aix.credentials.provider_env_vars(provider)

Return the env-var name(s) for `provider` as a list (possibly empty).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### Examples

```pycon
>>> provider_env_vars("openai")
['OPENAI_API_KEY']
>>> provider_env_vars("gemini")
['GEMINI_API_KEY', 'GOOGLE_API_KEY']
>>> provider_env_vars("totally-unknown-provider")
[]
```

### aix.credentials.requires_credentials(default_model)

Decorator: preflight credentials before the wrapped function runs.

Separates error-raising from business logic (per the AIX coding principles).
The wrapped function is expected to take keyword-only `model` and
`api_key` parameters. `default_model` is a zero-arg callable returning the
model the function would default to (read from the active config at call
time), so the preflight checks the *same* model the function will use.

* **Parameters:**
  **default_model** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[], [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Callable returning the default model id when the caller
  passes `model=None`.

### Examples

```pycon
>>> from aix import config
>>> @requires_credentials(lambda: config.get_config().chat.model)
... def f(*, model=None, api_key=None):
...     return "ran"
>>> f(api_key="sk-explicit")
'ran'
```

### aix.credentials.resolve_api_key(model_or_provider, , api_key=None, prompt_if_missing=False)

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
