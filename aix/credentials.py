"""Unified, discoverable API-key / secret resolution for AIX.

Every key-requiring function in AIX (``chat``, ``embeddings``, ``generate_image``,
``text_to_speech``, ``transcribe``, ``generate_video``, ...) resolves its
credentials through a single, documented path -- instead of relying on LiteLLM's
implicit environment reading, scattered ``os.getenv`` calls, or an orphaned
config getter.

Resolution layers, highest precedence first:

1. **Explicit argument** -- ``chat(..., api_key=...)``.
2. **Provider environment variable** -- the canonical name for the inferred
   provider (e.g. ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ``GEMINI_API_KEY``).
   A project ``.env`` is *softly* discovered when ``python-dotenv`` is installed
   (no hard dependency); explicit env vars always win over ``.env`` values.
3. **AIX config store** -- a config2py central store in the user app-config
   folder, keyed by the canonical env-var name (so the store and the env layer
   share one namespace). See :mod:`aix.util`.
4. **(REPL only) interactive prompt** -- when ``prompt_if_missing=True`` *and*
   running interactively, ask once and persist to the config store.

The provider for a model id is inferred via LiteLLM's ``get_llm_provider``; a
small curated :data:`PROVIDER_ENV_VARS` table maps each documented provider to
its env-var name(s) and console URL.

When a required key is genuinely absent, :func:`check_requirements` raises
:class:`MissingCredentialError` -- naming *which* key is missing, *how* to set
it, and *where* to obtain one -- rather than letting a cryptic, provider-specific
LiteLLM error surface late. This preflight is wired in via the
:func:`requires_credentials` decorator so error-raising stays separate from
business logic.

Examples:
    >>> from aix.credentials import resolve_api_key, PROVIDER_ENV_VARS
    >>> resolve_api_key("gpt-4o", api_key="sk-explicit")  # explicit wins
    'sk-explicit'
    >>> "openai" in PROVIDER_ENV_VARS
    True

Security:
    Resolved key *values* are never logged or returned by diagnostics;
    :func:`check_keys` reports availability only. Keys live in the user
    app-config folder, never in the repo.
"""

import os
from collections.abc import Iterable
from functools import wraps
from typing import Callable, Optional, Union

__all__ = [
    "PROVIDER_ENV_VARS",
    "PROVIDER_CONSOLE_URLS",
    "MissingCredentialError",
    "infer_provider",
    "provider_env_vars",
    "resolve_api_key",
    "check_requirements",
    "requires_credentials",
    "check_keys",
]


# --------------------------------------------------------------------------- #
# Provider <-> env-var mapping (curated table for the providers AIX documents)
# --------------------------------------------------------------------------- #

# Provider name (as inferred by ``litellm.get_llm_provider``) -> canonical env
# var name(s). When more than one is listed, they are tried in order; the first
# is treated as the *canonical* name for error messages and the config store.
PROVIDER_ENV_VARS: dict[str, Union[str, list[str]]] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    # LiteLLM uses GEMINI_API_KEY; the legacy google.generativeai SDK uses
    # GOOGLE_API_KEY -- accept either so both paths resolve the same secret.
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "vertex_ai": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    "openrouter": "OPENROUTER_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
    "groq": "GROQ_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "xai": "XAI_API_KEY",
    "perplexity": "PERPLEXITYAI_API_KEY",
    "together_ai": "TOGETHERAI_API_KEY",
    # Video providers (used by aix.video / aix.ai_models.sources).
    "runway": "RUNWAY_API_KEY",
    "pika": "PIKA_API_KEY",
}

# Provider -> where to obtain a key. Only the providers AIX documents are listed;
# a missing entry simply omits the "get one at" hint.
PROVIDER_CONSOLE_URLS: dict[str, str] = {
    "openai": "https://platform.openai.com/api-keys",
    "anthropic": "https://console.anthropic.com/settings/keys",
    "gemini": "https://aistudio.google.com/apikey",
    "vertex_ai": "https://console.cloud.google.com/apis/credentials",
    "google": "https://aistudio.google.com/apikey",
    "openrouter": "https://openrouter.ai/keys",
    "mistral": "https://console.mistral.ai/api-keys/",
    "cohere": "https://dashboard.cohere.com/api-keys",
    "groq": "https://console.groq.com/keys",
    "deepseek": "https://platform.deepseek.com/api_keys",
    "xai": "https://console.x.ai/",
    "perplexity": "https://www.perplexity.ai/settings/api",
    "runway": "https://app.runwayml.com/",
}


def provider_env_vars(provider: str) -> list[str]:
    """Return the env-var name(s) for ``provider`` as a list (possibly empty).

    Examples:
        >>> provider_env_vars("openai")
        ['OPENAI_API_KEY']
        >>> provider_env_vars("gemini")
        ['GEMINI_API_KEY', 'GOOGLE_API_KEY']
        >>> provider_env_vars("totally-unknown-provider")
        []
    """
    names = PROVIDER_ENV_VARS.get(provider)
    if names is None:
        return []
    return [names] if isinstance(names, str) else list(names)


# --------------------------------------------------------------------------- #
# Provider inference
# --------------------------------------------------------------------------- #


def infer_provider(model_or_provider: Optional[str]) -> Optional[str]:
    """Infer the provider name from a model id (or pass through a provider name).

    Uses LiteLLM's ``get_llm_provider`` when available. If ``model_or_provider``
    is already a known provider name (a key in :data:`PROVIDER_ENV_VARS`), it is
    returned as-is. Returns ``None`` if the provider cannot be determined.

    Examples:
        >>> infer_provider("gpt-4o")
        'openai'
        >>> infer_provider("openrouter")  # already a provider name
        'openrouter'
        >>> infer_provider(None) is None
        True
    """
    if not model_or_provider:
        return None
    # An explicit provider name short-circuits inference.
    if model_or_provider in PROVIDER_ENV_VARS:
        return model_or_provider
    try:
        from litellm import get_llm_provider

        # Returns (model, provider, dynamic_api_key, api_base).
        _, provider, *_ = get_llm_provider(model_or_provider)
        return provider or None
    except Exception:
        # LiteLLM missing, or it could not classify the id. Fall back to None;
        # the caller surfaces an actionable error rather than a cryptic one.
        return None


# --------------------------------------------------------------------------- #
# Store + .env access (non-interactive on the hot path)
# --------------------------------------------------------------------------- #

_dotenv_loaded = False


def _maybe_load_dotenv() -> None:
    """Softly load a project ``.env`` into ``os.environ`` (once), if dotenv exists.

    No-op when ``python-dotenv`` is not installed (it is not a hard dependency).
    Existing environment variables are never overridden, so an explicitly
    exported key always wins over a ``.env`` value.
    """
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True  # mark first so a missing dotenv is not retried
    try:
        from dotenv import load_dotenv, find_dotenv

        load_dotenv(find_dotenv(usecwd=True), override=False)
    except Exception:
        # dotenv absent or no .env found -- degrade to export-only discovery.
        pass


def _lookup_store(name: str) -> Optional[str]:
    """Read ``name`` from the AIX config2py store without prompting.

    Reads the central store directly (``get_config.configs``) rather than calling
    the composed getter, which may prompt interactively in a REPL. Returns
    ``None`` if absent or unreadable.
    """
    try:
        from aix.util import get_config as _aix_config_getter

        store = _aix_config_getter.configs
        value = store.get(name)
    except Exception:
        return None
    if value is None:
        return None
    value = value.strip()  # text-file stores can append a trailing newline
    return value or None


def _prompt_and_persist(name: str) -> Optional[str]:
    """Ask the user for ``name`` and persist it to the store (REPL only).

    Delegates to the config2py getter, which prompts and stores when interactive.
    Returns ``None`` if no value is obtained.
    """
    try:
        from aix.util import get_config as _aix_config_getter

        value = _aix_config_getter(name)
    except Exception:
        return None
    if not value:
        return None
    return value.strip() or None


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


def resolve_api_key(
    model_or_provider: Optional[str],
    *,
    api_key: Optional[str] = None,
    prompt_if_missing: bool = False,
) -> Optional[str]:
    """Resolve an API key for ``model_or_provider`` through the documented layers.

    Layers, highest precedence first: explicit ``api_key`` argument, provider
    environment variable (with soft ``.env`` discovery), the AIX config store,
    and -- only when ``prompt_if_missing`` is true *and* running interactively --
    an interactive prompt that persists the answer.

    Args:
        model_or_provider: A model id (provider inferred via LiteLLM) or a
            provider name directly (e.g. ``"openai"``).
        api_key: An explicit key; when given (non-empty) it is returned verbatim.
        prompt_if_missing: If true, fall back to a REPL prompt + persist when no
            key is found elsewhere. Off by default so the common path never
            blocks on input.

    Returns:
        The resolved key, or ``None`` if genuinely absent.

    Examples:
        >>> resolve_api_key("gpt-4o", api_key="sk-explicit")
        'sk-explicit'
    """
    # 1. Explicit argument always wins.
    if api_key:
        return api_key

    provider = infer_provider(model_or_provider)
    env_names = provider_env_vars(provider) if provider else []

    # 2. Provider environment variable (soft .env discovery first).
    _maybe_load_dotenv()
    for name in env_names:
        value = os.environ.get(name)
        if value:
            return value

    # 3. AIX config store (non-interactive read), keyed by env-var name.
    for name in env_names:
        value = _lookup_store(name)
        if value:
            return value

    # 4. (REPL only) prompt + persist, using the canonical env-var name.
    if prompt_if_missing and env_names:
        return _prompt_and_persist(env_names[0])

    return None


# --------------------------------------------------------------------------- #
# Preflight + informative errors
# --------------------------------------------------------------------------- #


class MissingCredentialError(Exception):
    """Raised when a required API key cannot be resolved.

    The message names which key is missing, how to set it, and (when known)
    where to obtain one. Key *values* are never included.
    """

    def __init__(
        self,
        model_or_provider: Optional[str],
        *,
        provider: Optional[str] = None,
        env_names: Optional[list[str]] = None,
    ):
        self.model_or_provider = model_or_provider
        self.provider = provider or infer_provider(model_or_provider)
        self.env_names = (
            env_names
            if env_names is not None
            else (provider_env_vars(self.provider) if self.provider else [])
        )
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        provider = self.provider or "the selected provider"
        if self.env_names:
            canonical = self.env_names[0]
            alt = (
                f" (or {', '.join(self.env_names[1:])})"
                if len(self.env_names) > 1
                else ""
            )
            how = (
                f"  - export it:  export {canonical}=...{alt}\n"
                f"  - add it to a .env file in your project, or\n"
                f"  - store it in the AIX config store under the key "
                f"{canonical!r}."
            )
            head = (
                f"No API key found for provider {provider!r} "
                f"(model {self.model_or_provider!r}). "
                f"Set {canonical}{alt} via one of:"
            )
        else:
            canonical = None
            how = (
                "  - pass it explicitly:  api_key=...\n"
                "  - or set the provider's documented environment variable."
            )
            head = (
                f"No API key found for {self.model_or_provider!r}, and the "
                f"provider could not be determined. Provide a key directly:"
            )
        url = PROVIDER_CONSOLE_URLS.get(self.provider or "")
        where = f"\nGet a key at: {url}" if url else ""
        return f"{head}\n{how}{where}"


def check_requirements(
    model_or_provider: Optional[str], *, api_key: Optional[str] = None
) -> bool:
    """Presence-only preflight: ensure a key for ``model_or_provider`` is resolvable.

    Does *not* validate the key over the network -- only that one is discoverable
    via :func:`resolve_api_key` (explicit arg, env/.env, or store). Raises
    :class:`MissingCredentialError` with actionable guidance when absent.

    Returns ``True`` when a key is available.

    Examples:
        >>> check_requirements("gpt-4o", api_key="sk-explicit")
        True
    """
    if resolve_api_key(model_or_provider, api_key=api_key) is not None:
        return True
    raise MissingCredentialError(model_or_provider)


def requires_credentials(default_model: Callable[[], Optional[str]]):
    """Decorator: preflight credentials before the wrapped function runs.

    Separates error-raising from business logic (per the AIX coding principles).
    The wrapped function is expected to take keyword-only ``model`` and
    ``api_key`` parameters. ``default_model`` is a zero-arg callable returning the
    model the function would default to (read from the active config at call
    time), so the preflight checks the *same* model the function will use.

    Args:
        default_model: Callable returning the default model id when the caller
            passes ``model=None``.

    Examples:
        >>> from aix import config
        >>> @requires_credentials(lambda: config.get_config().chat.model)
        ... def f(*, model=None, api_key=None):
        ...     return "ran"
        >>> f(api_key="sk-explicit")
        'ran'
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            model = kwargs.get("model") or default_model()
            check_requirements(model, api_key=kwargs.get("api_key"))
            return func(*args, **kwargs)

        return wrapper

    return decorator


# --------------------------------------------------------------------------- #
# Doctor / diagnostics
# --------------------------------------------------------------------------- #


def check_keys(providers: Optional[Iterable[str]] = None) -> dict[str, dict]:
    """Report, per provider, whether a usable API key is discoverable.

    Never returns or logs key *values* -- only availability and the env-var
    name(s) checked. Useful for quick setup debugging.

    Args:
        providers: Providers to check; defaults to every provider in
            :data:`PROVIDER_ENV_VARS`.

    Returns:
        Mapping ``provider -> {"available": bool, "env_vars": [...],
        "source": <where-found-or-None>}``. ``source`` is ``"env"``, ``"store"``,
        or ``None`` (never the value itself).

    Examples:
        >>> report = check_keys(["openai"])  # doctest: +SKIP
        >>> report["openai"]["available"]  # doctest: +SKIP
        True
    """
    names = list(providers) if providers is not None else list(PROVIDER_ENV_VARS)
    _maybe_load_dotenv()
    report: dict[str, dict] = {}
    for provider in names:
        env_names = provider_env_vars(provider)
        source = None
        for name in env_names:
            if os.environ.get(name):
                source = "env"
                break
        if source is None:
            for name in env_names:
                if _lookup_store(name):
                    source = "store"
                    break
        report[provider] = {
            "available": source is not None,
            "env_vars": env_names,
            "source": source,
        }
    return report
