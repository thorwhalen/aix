"""What a model supports, asked before calling it.

Providers differ in which OpenAI-style parameters they honour (``n``, ``seed``,
``response_format``...). LiteLLM knows, per model, through
``get_supported_openai_params``; this module puts a small, total (never
raising) question in front of it, so callers can *choose* a degraded path
rather than discover one silently.

Besides request parameter names, :func:`supports` answers a few *capabilities*
that are not a single parameter (see :data:`CAPABILITIES`), e.g.
``"response_schema"``: whether the model takes a JSON-Schema-constrained
``response_format`` (what ``aix.complete(..., response_schema=...)`` uses for
its ``"strict"`` / ``"schema"`` modes).

Why ask about ``seed`` in particular: a run-to-run difference with a seed the
provider *ignored* means nothing about model stability. ``supports(model,
"seed") is False`` rules a seed out; ``True`` only says LiteLLM forwards it --
seeded sampling is best-effort on every provider, and a router may drop it
upstream (compare
:attr:`aix.Completion.system_fingerprint` across runs: a change means the
serving backend changed, which voids the comparison).

>>> supports("gpt-4o-mini", "n")  # doctest: +SKIP
True
>>> supports("anthropic/claude-sonnet-4-20250514", "n")  # doctest: +SKIP
False
"""

from typing import Optional

from aix._litellm import UNRESOLVED as _UNRESOLVED, litellm_attr as _litellm_attr
from aix.config import get_config as _get_config, resolve_model as _resolve_model

__all__ = ["supports", "supported_params", "CAPABILITIES"]

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


def _supports_response_schema(model: str) -> bool:
    from aix.structured import model_supports_response_schema

    return model_supports_response_schema(model)


#: Capabilities :func:`supports` answers that are not one request parameter,
#: mapped to the ``model -> bool`` check behind each.
CAPABILITIES = {"response_schema": _supports_response_schema}


def supports(model: Optional[str], param: str) -> bool:
    """Whether ``model`` is known to accept ``param`` -- i.e. whether LiteLLM
    *forwards* it to the provider for this model.

    Conservative: ``False`` when support is unknown (or for a non-string
    ``param``), so a caller that branches on it takes the path that works
    everywhere. ``True`` is not a guarantee that the parameter has an effect:
    a router (OpenRouter...) can forward ``seed`` / ``n`` to an upstream model
    that drops them, and a provider can deprecate or ignore a parameter
    (seeding is best-effort everywhere; reasoning models may ignore it).

    Args:
        model: Model id or alias; ``None`` means the configured chat default.
        param: An OpenAI-style request parameter name (``"n"``, ``"seed"``,
            ``"response_format"``...) or a capability in :data:`CAPABILITIES`
            (``"response_schema"``).

    >>> supports("gpt-4o-mini", "seed")  # doctest: +SKIP
    True
    >>> supports("gpt-4o-mini", "response_schema")  # doctest: +SKIP
    True
    """
    if not isinstance(param, str):
        return False
    if param in CAPABILITIES:
        try:
            model = _resolve_model(model or _get_config().chat.model)
            return bool(CAPABILITIES[param](model))
        except Exception:  # total: a broken check means "not known to work"
            return False
    params = supported_params(model)
    return params is not None and param in params
