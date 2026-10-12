"""What a model supports, asked before calling it.

Providers differ in which OpenAI-style parameters they honour (``n``, ``seed``,
``response_format``...). LiteLLM knows, per model, through
``get_supported_openai_params``; this module puts a small, total (never
raising) question in front of it, so callers can *choose* a degraded path
rather than discover one silently.

>>> supports("gpt-4o-mini", "n")  # doctest: +SKIP
True
>>> supports("anthropic/claude-sonnet-4-20250514", "n")  # doctest: +SKIP
False
"""

from typing import Optional

from aix._litellm import UNRESOLVED as _UNRESOLVED, litellm_attr as _litellm_attr
from aix.config import get_config as _get_config, resolve_model as _resolve_model

__all__ = ["supports", "supported_params"]

# Deferred like every other litellm entry point (see aix/_litellm.py).
_litellm_get_supported_openai_params = _UNRESOLVED


def _get_supported_openai_params():
    global _litellm_get_supported_openai_params
    if _litellm_get_supported_openai_params is _UNRESOLVED:
        _litellm_get_supported_openai_params = _litellm_attr(
            "get_supported_openai_params"
        )
    return _litellm_get_supported_openai_params


def supported_params(model: Optional[str] = None) -> Optional[frozenset]:
    """The OpenAI-style parameters LiteLLM says ``model`` accepts.

    ``None`` when that is unknown (LiteLLM cannot place the model, or is not
    installed) -- which is not the same as "supports nothing".

    Args:
        model: Model id or alias; ``None`` means the configured chat default.
    """
    lookup = _get_supported_openai_params()
    if lookup is None:
        return None
    model = _resolve_model(model or _get_config().chat.model)
    try:
        params = lookup(model=model)
    except Exception:  # litellm raises on models it cannot route
        return None
    return frozenset(params) if params is not None else None


def supports(model: Optional[str], param: str) -> bool:
    """Whether ``model`` is known to accept the request parameter ``param``.

    Conservative: ``False`` when support is unknown, so a caller that branches
    on it takes the path that works everywhere.

    Args:
        model: Model id or alias; ``None`` means the configured chat default.
        param: An OpenAI-style request parameter name (``"n"``, ``"seed"``...).
    """
    params = supported_params(model)
    return params is not None and param in params
