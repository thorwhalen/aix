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

import hashlib
import numbers
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, TypeVar, Union

from aix.capabilities import supports
from aix.chat import Completion, Usage, complete
from aix.config import get_config as _get_config, submit_in_context
from aix.credentials import check_requirements as _check_requirements
from aix.structured import ResponseSchema, SchemaValidationError

__all__ = [
    "sample",
    "Sample",
    "Samples",
    "SamplingError",
    "concurrent_map",
    "check_positive_int",
]

T = TypeVar("T")
R = TypeVar("R")

Strategy = Literal["auto", "provider", "independent"]
_STRATEGIES = ("auto", "provider", "independent")
OnError = Literal["raise", "record"]
_ON_ERROR = ("raise", "record")

PROVIDER_N = "provider_n"
INDEPENDENT = "independent"
MIXED = "mixed"

# Request parameters ``sample`` owns; accepting them through ``**kwargs`` would
# silently change what is being sampled.
_RESERVED_KWARGS = frozenset({"messages", "stream"})
# Derived seeds stay in the non-negative int32 range every provider accepts.
_SEED_MASK = 0x7FFFFFFF


def check_positive_int(name: str, value: Any) -> int:
    """``int(value)`` if it is a positive integer (not a bool), else ``ValueError``.

    >>> check_positive_int("n", 3)
    3
    >>> check_positive_int("n", 0)
    Traceback (most recent call last):
    ...
    ValueError: n must be a positive integer, got 0
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def concurrent_map(
    func: Callable[[T], R], items: Iterable[T], *, concurrency: int
) -> list[R]:
    """``[func(x) for x in items]``, run on up to ``concurrency`` threads, in order.

    Fails fast like the serial loop would: once a call raises, calls not yet
    started are cancelled (those already running finish), and the first
    exception in item order propagates.

    >>> concurrent_map(lambda x: x * x, range(5), concurrency=3)
    [0, 1, 4, 9, 16]
    """
    items = list(items)
    if concurrency <= 1 or len(items) <= 1:
        return [func(x) for x in items]
    pool = ThreadPoolExecutor(max_workers=min(concurrency, len(items)))
    try:
        futures = [submit_in_context(pool, func, x) for x in items]
        return [f.result() for f in futures]
    finally:
        # On success every future is done; on failure, drop the queued ones.
        pool.shutdown(wait=True, cancel_futures=True)


def _derived_seed(seed: int, index: int) -> int:
    """A per-request seed from ``(seed, index)``, decorrelated across base seeds.

    Index 0 keeps the base seed, so ``sample(n=1, seed=s)`` reproduces
    ``complete(seed=s)``. Later indices hash: ``seed + index`` would make
    ``sample(seed=0)`` and ``sample(seed=1)`` share all but one request seed.

    >>> _derived_seed(42, 0)
    42
    >>> _derived_seed(0, 1) != _derived_seed(1, 1)
    True
    >>> _derived_seed(42, 3) == _derived_seed(42, 3)
    True
    """
    if index == 0:
        return seed
    digest = hashlib.blake2b(f"{seed}:{index}".encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big") & _SEED_MASK


@dataclass(frozen=True)
class Sample:
    """One drawn answer.

    Attributes:
        text: The raw text of this choice (``""`` if its request failed).
        payload: ``parse(text)`` when a ``parse`` function was given, else ``text``;
            ``None`` if the request or the parsing failed (see ``error``).
        finish_reason: Why this choice stopped. ``"length"`` = truncated,
            ``"content_filter"`` = refused/filtered; check it before treating the
            payload as a judgement.
        usage: Tokens for the request that produced *only this* sample, or
            ``None`` when it shared a provider-side ``n`` request (whose usage
            cannot be split per choice; see :attr:`Samples.usage`).
        strategy: How this sample was drawn: ``"provider_n"`` or ``"independent"``.
        seed: The seed sent with its request, if any.
        error: The exception this sample's request raised (with
            ``on_error="record"``) or that ``parse`` raised on its text.
        system_fingerprint: The serving backend's fingerprint for its request,
            when reported. Seeded samples are only comparable across equal
            fingerprints (see :func:`aix.supports` for whether ``seed`` is
            honoured at all).
    """

    text: str
    payload: Any
    finish_reason: Optional[str] = None
    usage: Optional[Usage] = None
    strategy: str = INDEPENDENT
    seed: Optional[int] = None
    error: Optional[Exception] = field(default=None, compare=False)
    system_fingerprint: Optional[str] = None

    @property
    def ok(self) -> bool:
        """``True`` when the request succeeded and the text parsed (if asked)."""
        return self.error is None


class Samples(list):
    """The list of :class:`Sample` returned by :func:`sample`, plus batch facts.

    It is a plain ``list`` otherwise: slicing or concatenating gives a ``list``
    without the batch attributes, which describe the whole draw only.

    Attributes:
        strategy: ``"provider_n"``, ``"independent"`` or ``"mixed"`` (provider
            ``n`` topped up with independent calls).
        usage: Total tokens over every request that reported usage, or ``None``
            if none did. Compare ``n_requests`` with ``n_usage_reports`` to see
            whether it is complete.
        n_requests: How many requests were made (schema repair requests included).
        n_usage_reports: How many of them reported usage (a completion's repair
            requests are counted with it when it reported usage). A provider
            request refused with a 400 (see ``fallback_error``) reports none.
        fallback_error: With ``strategy="auto"``: the error that made the
            provider-side ``n`` request fall back to independent requests
            (LiteLLM refusing ``n``, or the provider rejecting it with a 400),
            else ``None``.
    """

    def __init__(
        self,
        samples: Iterable[Sample] = (),
        *,
        strategy: str,
        usage: Optional[Usage] = None,
        n_requests: int = 0,
        n_usage_reports: int = 0,
        fallback_error: Optional[Exception] = None,
    ):
        super().__init__(samples)
        self.fallback_error = fallback_error
        self.strategy = strategy
        self.usage = usage
        self.n_requests = n_requests
        self.n_usage_reports = n_usage_reports

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple(s.text for s in self)

    @property
    def payloads(self) -> tuple[Any, ...]:
        return tuple(s.payload for s in self)

    @property
    def errors(self) -> tuple[Exception, ...]:
        """Every recorded error (request or parse), in sample order."""
        return tuple(s.error for s in self if s.error is not None)

    def __repr__(self) -> str:
        return (
            f"Samples({list.__repr__(self)}, strategy={self.strategy!r}, "
            f"usage={self.usage!r}, n_requests={self.n_requests}"
            + (
                f", fallback_error={self.fallback_error!r})"
                if self.fallback_error is not None
                else ")"
            )
        )


class SamplingError(Exception):
    """A request inside :func:`sample` failed (with ``on_error="raise"``).

    No request is launched after the first failure, and the ones already in
    flight are awaited, so the work already paid for is not lost: ``samples``
    holds every request that was made, failed ones included (as samples whose
    ``error`` is set) -- possibly fewer than ``n``. The first failure is
    ``__cause__``.
    """

    def __init__(self, message: str, *, samples: Optional[Samples] = None):
        # ``samples`` has a default so the exception survives pickling
        # (multiprocessing / joblib re-create it from ``args`` alone).
        super().__init__(message)
        self.samples = samples


def _parsed(text: str, parse: Optional[Callable[[str], Any]]):
    """``(payload, error)`` for ``text``."""
    if parse is None:
        return text, None
    try:
        return parse(text), None
    except Exception as e:  # one unparsable sample must not sink the batch
        return None, e


def _is_request_error(error: Optional[Exception]) -> bool:
    """Whether a recorded per-choice error is a failed request (e.g. a repair
    request that raised), as opposed to a reply that did not parse/validate."""
    return error is not None and not isinstance(error, (SchemaValidationError,))


def _payload(c: Completion, index: int, parse: Optional[Callable[[str], Any]]):
    """``(payload, error)`` for choice ``index`` of ``c``.

    The validated value when ``c`` was produced with a response schema, else
    ``parse`` applied to the text.
    """
    if c.schema_mode is not None:
        if index >= len(c.values):
            return None, ValueError("The provider returned no choice for this sample")
        return c.values[index], c.errors[index]
    return _parsed(c.texts[index] if index < len(c.texts) else "", parse)


def _has_class_named(error: Exception, name: str) -> bool:
    """Whether ``error`` is an instance of a class called ``name``.

    Matched by name: aix must not import litellm just to name its exceptions.
    """
    return any(cls.__name__ == name for cls in type(error).__mro__)


def _is_unsupported_param_error(error: Exception) -> bool:
    """LiteLLM refused a request parameter client-side, before sending it."""
    return _has_class_named(error, "UnsupportedParamsError")


# 400 subclasses that are never about ``n``: resending the prompt alone would
# only pay for the same rejection twice.
_NOT_ABOUT_N_ERRORS = (
    "ContextWindowExceededError",
    "ContentPolicyViolationError",
    "RejectedRequestError",
)


def _is_bad_request_error(error: Exception) -> bool:
    """The provider rejected the request (HTTP 400) -- possibly because of ``n``."""
    return _has_class_named(error, "BadRequestError") and not any(
        _has_class_named(error, name) for name in _NOT_ABOUT_N_ERRORS
    )


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
    response_schema: Union[dict, type] = None,
    on_error: OnError = "raise",
    api_key: str = None,
    **kwargs,
) -> Samples:
    """Draw ``n`` answers to ``prompt``, each with its finish reason and usage.

    Args:
        prompt: A string or a list of message dicts, as for :func:`aix.chat`.
        n: How many samples to draw.
        temperature: Sampling temperature (``None`` = configured default). At
            ``0`` the samples mostly coincide, which is rarely what you want here.
        seed: Base seed. A provider-side ``n`` request sends it as is; each
            independent request sends a seed derived from ``(seed, index)`` so
            independent samples are not forced identical and runs with different
            base seeds do not overlap. Support is provider-dependent.
        concurrency: Max simultaneous independent requests (``None`` = config
            ``chat.sample_concurrency``).
        model: Model id or alias (``None`` = configured default).
        strategy: ``"auto"`` (provider ``n`` when supported, else independent;
            also falls back to independent if LiteLLM or the provider refuses
            ``n`` -- for a provider 400 the first independent request is sent
            alone as a probe, and if it fails too the original 400 is raised;
            see :attr:`Samples.fallback_error`),
            ``"provider"`` (force one ``n`` request) or ``"independent"``.
            ``n=1`` is always a single plain request, recorded ``"independent"``.
        response_schema: A JSON Schema dict or Pydantic model class, enforced and
            validated per request as in :func:`aix.complete`; each payload is the
            validated value, and a choice still invalid after repairs records its
            ``SchemaValidationError`` as ``error``. Exclusive with ``parse``.
        parse: Optional ``text -> payload`` (e.g. ``json.loads``). A failure is
            recorded on that sample (``payload=None``, ``error=...``), not raised.
        on_error: When a request fails: ``"raise"`` (default) raises
            :class:`SamplingError` once the launched requests finish, carrying
            the partial batch; ``"record"`` returns the batch with the failure on
            that sample's ``error``.
        api_key: Explicit API key, as for :func:`aix.chat`.
        **kwargs: Passed to every request (e.g. ``max_tokens``, ``response_format``).

    Returns:
        :class:`Samples` -- a list of :class:`Sample` with batch ``.strategy``,
        ``.usage``, ``.n_requests``.

    Raises:
        ValueError: For a non-positive ``n`` / ``concurrency``, an unknown
            ``strategy`` / ``on_error``, or a reserved kwarg (``messages``, ``stream``).
        SamplingError: A request failed and ``on_error="raise"``.
    """
    n = check_positive_int("n", n)
    if concurrency is None:
        concurrency = _get_config().chat.sample_concurrency
    concurrency = check_positive_int("concurrency", concurrency)
    if strategy not in _STRATEGIES:
        raise ValueError(f"strategy must be one of {_STRATEGIES}, got {strategy!r}")
    if on_error not in _ON_ERROR:
        raise ValueError(f"on_error must be one of {_ON_ERROR}, got {on_error!r}")
    if parse is not None and response_schema is not None:
        raise ValueError("Pass parse or response_schema, not both.")
    reserved = _RESERVED_KWARGS.intersection(kwargs)
    if reserved:
        raise ValueError(
            f"sample() does not accept {sorted(reserved)}: pass the conversation "
            "as `prompt`; sampling does not stream."
        )

    # Preflight credentials once, here, so a missing key raises itself instead
    # of n copies of it surfacing from the pool as a SamplingError.
    _check_requirements(model or _get_config().chat.model, api_key=api_key)

    call_kwargs = dict(model=model, temperature=temperature, api_key=api_key, **kwargs)
    if response_schema is not None:
        # Normalized once here, not once per request (it builds a validator).
        call_kwargs["response_schema"] = ResponseSchema.from_schema(response_schema)

    samples: list[Sample] = []
    usages: list[Usage] = []
    n_requests = n_usage_reports = 0
    # A failed request -- including a schema repair request -- as opposed to a
    # reply that did not parse/validate (which is always just recorded).
    first_error, n_failed = None, 0

    use_provider_n = n > 1 and (
        strategy == "provider" or (strategy == "auto" and supports(model, "n"))
    )
    # Why the provider-n request was abandoned for independent requests, if it was.
    fallback_error: Optional[Exception] = None
    # A 400 may or may not be about ``n``: the first independent request is
    # then sent alone, as a probe, before the rest are launched.
    probe_first = False
    if use_provider_n:
        try:
            batch = complete(prompt, n=n, seed=seed, **call_kwargs)
        except Exception as e:
            if strategy != "auto":
                raise
            if _is_unsupported_param_error(e):
                pass  # LiteLLM knows n is unsupported: supports() was wrong
            elif _is_bad_request_error(e):
                probe_first = True  # the provider refused it server-side
                n_requests += 1  # unlike a LiteLLM refusal, this one was sent
                n_failed += 1
            else:
                raise
            use_provider_n, fallback_error = False, e
        else:
            n_requests += 1 + batch.repairs
            for i, (text, reason) in enumerate(
                list(zip(batch.texts, batch.finish_reasons))[:n]
            ):
                payload, error = _payload(batch, i, parse)
                if batch.schema_mode is not None and _is_request_error(error):
                    first_error, n_failed = first_error or error, n_failed + 1
                samples.append(
                    Sample(
                        text,
                        payload,
                        reason,
                        None,
                        PROVIDER_N,
                        seed,
                        error,
                        system_fingerprint=batch.system_fingerprint,
                    )
                )
            if batch.usage is not None:
                usages.append(batch.usage)
                n_usage_reports += 1 + batch.repairs

    n_from_provider = len(samples)
    seeds = [
        None if seed is None else _derived_seed(seed, j)
        for j in range(n_from_provider, n)
    ]

    # With on_error="raise", stop launching once a request has failed (the
    # batch is lost anyway); requests already in flight still finish.
    stop = threading.Event()

    def draw_one(one_seed: Optional[int]) -> Union[Completion, Exception, None]:
        if stop.is_set():
            return None  # never launched
        try:
            return complete(prompt, n=1, seed=one_seed, **call_kwargs)
        except Exception as e:  # collected, so finished work is not discarded
            if on_error == "raise":
                stop.set()
            return e

    results: list = []
    if probe_first and seeds:
        probe = draw_one(seeds[0])
        if isinstance(probe, Exception) and _is_bad_request_error(probe):
            # A plain request is rejected too: the 400 was not about n. Raise
            # the probe's own rejection (the real reason), chained to the first.
            raise probe from fallback_error
        # Success, or an ordinary failure (rate limit, timeout...): from here
        # the probe is just the first independent request (on_error applies).
        results.append(probe)
    results += concurrent_map(draw_one, seeds[len(results) :], concurrency=concurrency)

    for one_seed, c in zip(seeds, results):
        if c is None:
            continue
        n_requests += 1 + (0 if isinstance(c, Exception) else c.repairs)
        if isinstance(c, Exception):
            first_error, n_failed = first_error or c, n_failed + 1
            samples.append(Sample("", None, None, None, INDEPENDENT, one_seed, c))
            continue
        payload, error = _payload(c, 0, parse)
        if c.schema_mode is not None and _is_request_error(error):
            first_error, n_failed = first_error or error, n_failed + 1
        samples.append(
            Sample(
                c.text,
                payload,
                c.finish_reason,
                c.usage,
                INDEPENDENT,
                one_seed,
                error,
                system_fingerprint=c.system_fingerprint,
            )
        )
        if c.usage is not None:
            usages.append(c.usage)
            n_usage_reports += 1 + c.repairs

    if not use_provider_n:
        recorded = INDEPENDENT
    else:
        recorded = MIXED if n_from_provider < n else PROVIDER_N
    result = Samples(
        samples,
        strategy=recorded,
        usage=sum(usages, Usage()) if usages else None,
        n_requests=n_requests,
        n_usage_reports=n_usage_reports,
        fallback_error=fallback_error,
    )
    if first_error is not None and on_error == "raise":
        raise SamplingError(
            f"{n_failed} of {n_requests} sampling request(s) failed; "
            "the partial batch is on .samples",
            samples=result,
        ) from first_error
    return result
