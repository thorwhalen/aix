# aix

AIX: Artificial Intelligence eXtensions — a clean, pythonic facade over `litellm`
for chat, embeddings, prompt-functions, image/audio/video generation and model
discovery, abstracting away provider-specific details.

## Module map (`aix/`)

- `chat.py`, `embeddings.py`, `image.py`, `audio.py`, `video.py`, `vision.py` —
  one facade module per capability (chat completions, vector embeddings,
  text->image, TTS/transcription, text/image->video, image->text).
- `models.py` + `ai_models/` — model discovery/selection (`ai_models/sources.py`,
  `manager.py`, `base.py`, `examples.py`).
- `prompts.py` — `prompt_func`: turn a text prompt into a callable Python function.
- `batches.py` — batch processing of prompts/embeddings.
- `config.py` — **SSOT** for per-operation default model/params.
- `credentials.py` — unified, discoverable API-key resolution used by every
  key-requiring function; raises `MissingCredentialError` when a key is absent.
- `_litellm.py` — deferred access to the `litellm` backend: `aix` never imports
  `litellm` at its own import time (see Invariants).
- `gen_ai/` — legacy provider-specific backends (`openai_genai.py`, `google_genai.py`).
- `stores.py` — storage layers (marked TODO/incomplete in-repo).
- `misc.py`, `util.py` — small helpers (e.g. `misc.get_llm_leaderboards`).

## Tests & lint (verified)

```bash
uv venv .venv && uv pip install -e . pytest pytest-cov pytest-mock ruff
.venv/bin/pytest tests/ -q                              # 277 passed, fully mocked/hermetic
.venv/bin/pytest --doctest-modules aix/ --ignore=aix/gen_ai/ -q
.venv/bin/ruff check .
```
`run_tests.sh` wraps both and deliberately does not fail the build on doctest
errors — some doctests call real network (`OpenRouterSource.discover_models`) or
need optional extras not installed by the bare package (e.g. `numpy` in an
`embeddings.py` example) and are expected to fail in a minimal/offline env.

## Invariants

- **`import aix` must never touch the network.** `litellm` fetches its model-cost
  map over HTTPS as an import-time side effect; `aix._litellm` defers that import
  (issue #38). `tests/test_import_is_hermetic.py` guards this in a subprocess with
  a socket guard — do not weaken or bypass it when adding a new provider backend.
- The root `conftest.py` (not `tests/conftest.py`) sets
  `LITELLM_LOCAL_MODEL_COST_MAP=1` so it also covers the `--doctest-modules`
  collection tree; `tests/conftest.py` seeds dummy provider keys for the whole
  suite (LiteLLM itself is mocked in tests).
- `config.py` is the single source of truth for per-operation model defaults —
  don't hardcode a model name elsewhere.

## Docs

- `examples/` — one runnable script per capability (01-09).
- `misc/docs/` — `aix_resources_summary.md`, `implementation.md`.
- `aix/ai_models/README.md` — model-discovery subsystem notes.

## Dependents

`mood`, `newsmood` import this package — check their tests before changing any
public facade function's signature or return shape.
