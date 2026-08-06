"""Deferred access to the ``litellm`` backend.

``aix`` never imports ``litellm`` at *its own* import time, because importing
``litellm`` is not free:

* it costs roughly a second of CPU on its own, and
* **as a side effect of the import itself** it fetches its model-cost map over
  HTTPS from a raw-content host (5 s timeout, silent fallback to a bundled
  copy on any failure).

``aix`` is a facade that downstream projects import at process start, so paying
that on ``import aix`` makes every consumer inherit an unbounded outbound
request before doing any work: a server cold start blocks on it, an offline or
restricted-egress deployment eats a connect timeout, and a test suite that only
does ``importorskip("aix.audio")`` reaches the network.  See
https://github.com/thorwhalen/aix/issues/38.

The fix is *when*, not *whether*.  Each ``aix`` module binds its litellm entry
points to the :data:`UNRESOLVED` sentinel and resolves them through
:func:`litellm_attr` on first use, so the import — and therefore the cost-map
fetch — happens on the first actual provider call.  That call is already going
over the network, so the map is still the fresh remote one and provider
resolution is unchanged.

Why not force litellm's bundled map instead
-------------------------------------------
``litellm`` honours ``LITELLM_LOCAL_MODEL_COST_MAP``: set it to any non-empty
value and litellm reads a copy bundled in the wheel instead of fetching.  It is
tempting for ``aix`` to set that itself, but it is the wrong default here:

* **It is not free.** ``aix`` reads none of litellm's cost data (model pricing
  in :mod:`aix.ai_models` comes from provider APIs), but the cost map also
  backs litellm's per-provider model lists, which is how a *bare* model name
  such as ``"gpt-5.2"`` is routed to a provider.  The bundled copy is a
  snapshot: at the time of writing it carried 1911 entries against the remote
  map's 2988, so bare names newer than the installed ``litellm`` release stop
  resolving.  (Provider-prefixed names such as ``"openai/gpt-5.2"`` resolve
  either way.)
* **It does not fix the import cost**, only the request — deferring the import
  fixes both.
* **It is a whole-process mutation.** ``os.environ`` is global, so a library
  setting it changes the behaviour of every other litellm user in the process.

So ``aix`` sets no environment variable.  Operators who want the first call to
be offline too can export ``LITELLM_LOCAL_MODEL_COST_MAP=1`` themselves, with
the narrower bare-name resolution above as the known trade-off; ``aix`` stays
out of that decision.
"""

from typing import Any

__all__ = ["UNRESOLVED", "litellm_attr"]


class _Unresolved:
    """Type of the :data:`UNRESOLVED` sentinel."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNRESOLVED"

    def __bool__(self) -> bool:
        raise TypeError(
            "UNRESOLVED is a 'not looked up yet' marker and has no truth value. "
            "Resolve it through the module's loader (see aix._litellm)."
        )


#: Marker for "this litellm entry point has not been looked up yet".
#:
#: Distinct from ``None``, which means "looked up, and litellm is not
#: installed" — the state the call sites turn into an actionable
#: :class:`ImportError`.  It is deliberately not falsy: a stray
#: ``if _litellm_completion:`` would otherwise silently take the
#: not-installed branch for an entry point that is merely unresolved.
UNRESOLVED: Any = _Unresolved()


def litellm_attr(name: str) -> Any:
    """Import ``litellm`` and return the attribute *name*, or ``None`` if absent.

    ``None`` is returned both when ``litellm`` is not installed and when it is
    installed but does not expose *name* (an entry point added or removed
    across litellm versions).  Both mean the same thing to a caller: this
    backend cannot serve the request, so raise an install hint.

    Callers are expected to cache the result in a module global so the import
    is attempted once per entry point.  That cache needs no lock: ``import`` is
    already serialised per module by the interpreter, and two threads racing to
    resolve the same entry point compute and store the *same* object, so the
    worst case is a duplicated ``getattr`` (:mod:`aix.batches` calls these from
    a thread pool, so the race is real, not hypothetical).

    >>> callable(litellm_attr("completion")) or litellm_attr("completion") is None
    True
    >>> litellm_attr("no_such_litellm_entry_point_") is None
    True
    """
    try:
        import litellm
    except ImportError:  # pragma: no cover - exercised only without litellm
        return None
    return getattr(litellm, name, None)
