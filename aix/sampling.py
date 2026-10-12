"""Repeated sampling: ``n`` answers to one prompt, drawn as fast as the provider allows.

Asking the same question several times is the basic variance-reduction (and
stability-measurement) tool for LLM judgements. :func:`sample` does it in one of
two ways and **records which**, because they are not the same statistical object:

* ``"provider_n"`` -- one request with ``n`` choices (one prompt evaluation, ``n``
  decodes). Cheapest, and used when LiteLLM reports the model supports ``n``.
* ``"independent"`` -- ``n`` separate requests, run concurrently. Works with every
  provider; each sample also gets its own ``usage``.

If a provider silently returns fewer choices than asked, the shortfall is drawn
independently and the batch is marked ``"mixed"``.

>>> answers = sample("Is a tomato a fruit? Answer yes or no.", n=5)  # doctest: +SKIP
>>> answers.texts  # doctest: +SKIP
('Yes', 'Yes', 'Yes', 'yes', 'Yes')
>>> answers.strategy, answers.usage.total_tokens  # doctest: +SKIP
('provider_n', 52)
"""

import numbers
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, TypeVar, Union

from aix.capabilities import supports
from aix.chat import Completion, Usage, complete
from aix.config import get_config as _get_config

__all__ = ["sample", "Sample", "Samples", "concurrent_map"]

T = TypeVar("T")
R = TypeVar("R")

Strategy = Literal["auto", "provider", "independent"]
_STRATEGIES = ("auto", "provider", "independent")

PROVIDER_N = "provider_n"
INDEPENDENT = "independent"
MIXED = "mixed"


def concurrent_map(
    func: Callable[[T], R], items: Iterable[T], *, concurrency: int
) -> list[R]:
    """``[func(x) for x in items]``, run on up to ``concurrency`` threads, in order.

    The first exception (in item order) propagates, as it would serially.

    >>> concurrent_map(lambda x: x * x, range(5), concurrency=3)
    [0, 1, 4, 9, 16]
    """
    items = list(items)
    if concurrency <= 1 or len(items) <= 1:
        return [func(x) for x in items]
    with ThreadPoolExecutor(max_workers=min(concurrency, len(items))) as pool:
        return list(pool.map(func, items))


@dataclass(frozen=True)
class Sample:
    """One drawn answer.

    Attributes:
        text: The raw text of this choice.
        payload: ``parse(text)`` when a ``parse`` function was given, else ``text``;
            ``None`` if parsing failed (see ``error``).
        finish_reason: Why this choice stopped. ``"length"`` = truncated,
            ``"content_filter"`` = refused/filtered; check it before treating the
            payload as a judgement.
        usage: Tokens for the request that produced *only this* sample, or
            ``None`` when it shared a provider-side ``n`` request (whose usage
            cannot be split per choice; see :attr:`Samples.usage`).
        strategy: How this sample was drawn: ``"provider_n"`` or ``"independent"``.
        seed: The seed sent with its request, if any.
        error: The exception ``parse`` raised on this text, if it did.
    """

    text: str
    payload: Any
    finish_reason: Optional[str] = None
    usage: Optional[Usage] = None
    strategy: str = INDEPENDENT
    seed: Optional[int] = None
    error: Optional[Exception] = field(default=None, compare=False)

    @property
    def ok(self) -> bool:
        """``True`` when the text was parsed (or no parsing was asked for)."""
        return self.error is None


class Samples(list):
    """The list of :class:`Sample` returned by :func:`sample`, plus batch facts.

    Attributes:
        strategy: ``"provider_n"``, ``"independent"`` or ``"mixed"`` (provider
            ``n`` topped up with independent calls).
        usage: Total tokens over every request made, or ``None`` if no request
            reported usage.
    """

    def __init__(
        self,
        samples: Iterable[Sample] = (),
        *,
        strategy: str,
        usage: Optional[Usage] = None,
    ):
        super().__init__(samples)
        self.strategy = strategy
        self.usage = usage

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple(s.text for s in self)

    @property
    def payloads(self) -> tuple[Any, ...]:
        return tuple(s.payload for s in self)

    def __repr__(self) -> str:
        return (
            f"Samples({list.__repr__(self)}, strategy={self.strategy!r}, "
            f"usage={self.usage!r})"
        )


def _check_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _parsed(text: str, parse: Optional[Callable[[str], Any]]):
    """``(payload, error)`` for ``text``."""
    if parse is None:
        return text, None
    try:
        return parse(text), None
    except Exception as e:  # one unparsable sample must not sink the batch
        return None, e


def sample(
    prompt: Union[str, Iterable[dict]],
    *,
    n: int,
    temperature: float = None,
    seed: int = None,
    concurrency: int = None,
    model: str = None,
    strategy: Strategy = "auto",
    parse: Callable[[str], Any] = None,
    api_key: str = None,
    **kwargs,
) -> Samples:
    """Draw ``n`` answers to ``prompt``, each with its finish reason and usage.

    Args:
        prompt: A string or a list of message dicts, as for :func:`aix.chat`.
        n: How many samples to draw.
        temperature: Sampling temperature (``None`` = configured default). At
            ``0`` the samples mostly coincide, which is rarely what you want here.
        seed: Base seed. A provider-side ``n`` request sends it as is; the
            ``j``-th independent request sends ``seed + j`` (plus one if a
            provider request came first) so independent samples are not forced
            to be identical. Support is provider-dependent.
        concurrency: Max simultaneous independent requests (``None`` = config
            ``chat.sample_concurrency``).
        model: Model id or alias (``None`` = configured default).
        strategy: ``"auto"`` (provider ``n`` when supported, else independent),
            ``"provider"`` (force one ``n`` request) or ``"independent"``.
        parse: Optional ``text -> payload`` (e.g. ``json.loads``). A failure is
            recorded on that sample (``payload=None``, ``error=...``), not raised.
        api_key: Explicit API key, as for :func:`aix.chat`.
        **kwargs: Passed to every request (e.g. ``max_tokens``, ``response_format``).

    Returns:
        :class:`Samples` -- a list of :class:`Sample` with ``.strategy`` and
        ``.usage`` (the batch total).

    Raises:
        ValueError: For a non-positive ``n`` / ``concurrency`` or an unknown
            ``strategy``. Request errors propagate.
    """
    n = _check_positive_int("n", n)
    if concurrency is None:
        concurrency = _get_config().chat.sample_concurrency
    concurrency = _check_positive_int("concurrency", concurrency)
    if strategy not in _STRATEGIES:
        raise ValueError(f"strategy must be one of {_STRATEGIES}, got {strategy!r}")

    call_kwargs = dict(model=model, temperature=temperature, api_key=api_key, **kwargs)

    use_provider_n = strategy == "provider" or (
        strategy == "auto" and n > 1 and supports(model, "n")
    )

    samples: list[Sample] = []
    usages: list[Usage] = []
    if use_provider_n:
        batch = complete(prompt, n=n, seed=seed, **call_kwargs)
        for text, reason in list(zip(batch.texts, batch.finish_reasons))[:n]:
            payload, error = _parsed(text, parse)
            samples.append(Sample(text, payload, reason, None, PROVIDER_N, seed, error))
        if batch.usage is not None:
            usages.append(batch.usage)

    missing = n - len(samples)
    offset = 1 if use_provider_n else 0
    seeds = [None if seed is None else seed + offset + j for j in range(missing)]

    def draw_one(one_seed: Optional[int]) -> Completion:
        return complete(prompt, n=1, seed=one_seed, **call_kwargs)

    for one_seed, c in zip(
        seeds, concurrent_map(draw_one, seeds, concurrency=concurrency)
    ):
        payload, error = _parsed(c.text, parse)
        samples.append(
            Sample(
                c.text, payload, c.finish_reason, c.usage, INDEPENDENT, one_seed, error
            )
        )
        if c.usage is not None:
            usages.append(c.usage)

    if not use_provider_n:
        recorded = INDEPENDENT
    else:
        recorded = MIXED if missing else PROVIDER_N
    return Samples(
        samples,
        strategy=recorded,
        usage=sum(usages, Usage()) if usages else None,
    )
