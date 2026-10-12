"""Core chat interface for AIX.

A clean, pythonic facade for chat completions across multiple AI providers.
Uses LiteLLM as the backend but provides a simpler, more intuitive interface.

Examples:
    Simple text prompt:
    >>> from aix.chat import chat
    >>> response = chat("What is 2+2?")  # doctest: +SKIP
    >>> print(response)  # doctest: +SKIP
    'The answer is 4.'

    With specific model:
    >>> chat("Hello!", model="gpt-4o-mini")  # doctest: +SKIP
    'Hello! How can I help you today?'

    With message history:
    >>> messages = [
    ...     {"role": "user", "content": "My name is Alice"},
    ...     {"role": "assistant", "content": "Nice to meet you, Alice!"},
    ...     {"role": "user", "content": "What's my name?"}
    ... ]
    >>> chat(messages, model="gpt-4o-mini")  # doctest: +SKIP
    'Your name is Alice.'

    Streaming responses:
    >>> for chunk in chat("Count to 3", stream=True):  # doctest: +SKIP
    ...     print(chunk, end='')
    1, 2, 3

    The whole response -- every choice, token usage, finish reasons, the model
    that actually served the call -- via :func:`complete`:

    >>> c = complete("Name a color", n=3, temperature=1.0)  # doctest: +SKIP
    >>> c.texts  # doctest: +SKIP
    ('Blue', 'Green', 'Red')
    >>> c.usage.total_tokens, c.finish_reasons  # doctest: +SKIP
    (41, ('stop', 'stop', 'stop'))
"""

from collections.abc import Iterable, Mapping
import numbers
from functools import wraps
from dataclasses import (
    dataclass,
    field,
    fields as dataclass_fields,
    replace as _replace_dataclass,
)
from typing import Union, Any, Optional

from aix.config import (
    get_config as _get_config,
    resolve_model as _resolve_model,
    ChatConfig as _ChatConfig,
)
from aix.credentials import (
    resolve_api_key as _resolve_api_key,
    requires_credentials as _requires_credentials,
)

from aix._litellm import UNRESOLVED as _UNRESOLVED, litellm_attr as _litellm_attr
from aix.structured import (
    ResponseSchema as _ResponseSchema,
    JSON_MODE as _JSON_MODE,
    PROMPT as _PROMPT,
)

# Modes in which the provider is not told the schema, so the prompt must say it.
_SCHEMA_IN_PROMPT_MODES = (_JSON_MODE, _PROMPT)
# Assistant turn standing in for an empty reply in a repair request (some
# providers reject empty text blocks).
_EMPTY_REPLY_PLACEHOLDER = "(empty reply)"

# LiteLLM is kept private (users shouldn't call it directly) *and* deferred:
# importing it fetches litellm's model-cost map over HTTPS, so `import aix`
# must not do it. See aix/_litellm.py for the full rationale.
_litellm_completion = _UNRESOLVED


def _completion():
    """Resolve litellm's ``completion``, importing litellm on first use.

    Returns ``None`` when litellm is unavailable, which call sites turn into an
    :class:`ImportError` carrying an install hint.
    """
    global _litellm_completion
    if _litellm_completion is _UNRESOLVED:
        _litellm_completion = _litellm_attr("completion")
    return _litellm_completion


# Shipped-default constants, kept for backward compatibility. The *active*
# defaults are resolved from ``aix.config`` at call time (see aix/config.py),
# so changing config -- not these constants -- is what affects behavior.
DFLT_CHAT_MODEL = _ChatConfig().model
DFLT_TEMPERATURE = _ChatConfig().temperature
DFLT_MAX_TOKENS = _ChatConfig().max_tokens


def _normalize_prompt(
    prompt: Union[str, Iterable[dict]],
) -> list[dict]:
    """Normalize prompt input to standard message format.

    Args:
        prompt: Either a string or list of message dicts

    Returns:
        List of message dictionaries with 'role' and 'content'

    Examples:
        >>> _normalize_prompt("Hello")
        [{'role': 'user', 'content': 'Hello'}]

        >>> _normalize_prompt([{"role": "user", "content": "Hi"}])
        [{'role': 'user', 'content': 'Hi'}]
    """
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    elif isinstance(prompt, Iterable):
        # Convert to list if it's an iterable
        messages = list(prompt)
        # Validate structure
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ValueError(
                    f"Invalid message format: {msg}. "
                    "Messages must be dicts with 'role' and 'content' keys."
                )
        return messages
    else:
        raise TypeError(
            f"Prompt must be a string or iterable of message dicts, got {type(prompt)}"
        )


def _extract_text_from_response(response: Any) -> str:
    """Extract text content from LiteLLM response.

    Args:
        response: LiteLLM ModelResponse object

    Returns:
        Extracted text content
    """
    try:
        return response.choices[0].message.content or ""
    except (AttributeError, IndexError, KeyError):
        # Fallback for unexpected response structure
        return str(response)


def _extract_text_from_stream(stream: Iterable) -> Iterable[str]:
    """Extract text chunks from LiteLLM streaming response.

    Args:
        stream: LiteLLM streaming response

    Yields:
        Text chunks as they arrive
    """
    for chunk in stream:
        try:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content is not None:
                yield delta.content
        except (AttributeError, IndexError, KeyError):
            # Skip malformed chunks
            continue


def _field(obj: Any, name: str) -> Any:
    """Read ``name`` from a LiteLLM response part, attribute- or key-style.

    LiteLLM returns pydantic-ish objects, but some providers and caches hand
    back plain dicts; reading both keeps extraction provider-agnostic.
    """
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _as_int(value: Any) -> Optional[int]:
    """``value`` as an integer count if it is one (``7`` or ``7.0``), else ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _as_str(value: Any) -> Optional[str]:
    """``value`` if it is a string, else ``None``."""
    return value if isinstance(value, str) else None


def _content_text(content: Any) -> str:
    """The text of a message ``content``: a string, or the text parts of a list.

    Multimodal / structured providers return content as a list of parts
    (``[{"type": "text", "text": "..."}, ...]``); their text parts are joined.
    Anything else (``None`` for a pure tool call) is ``""``.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = (_as_str(_field(part, "text")) for part in content)
        return "".join(p for p in parts if p)
    return ""


def _nested(obj: Any, path: str) -> Any:
    """Follow a dotted ``path`` of :func:`_field` reads, ``None`` if any is absent."""
    for name in path.split("."):
        if obj is None:
            return None
        obj = _field(obj, name)
    return obj


# Usage field -> where LiteLLM puts it, first match wins. LiteLLM normalizes most
# providers to the OpenAI shape (``*_tokens_details``); the bare Anthropic names
# are the fallback for responses it passes through unnormalized.
_USAGE_SOURCES = {
    "prompt_tokens": ("prompt_tokens",),
    "completion_tokens": ("completion_tokens",),
    "total_tokens": ("total_tokens",),
    "cached_prompt_tokens": (
        "prompt_tokens_details.cached_tokens",
        "cache_read_input_tokens",
    ),
    "cache_creation_tokens": (
        "prompt_tokens_details.cache_creation_tokens",
        "cache_creation_input_tokens",
    ),
    "reasoning_tokens": ("completion_tokens_details.reasoning_tokens",),
}


@dataclass(frozen=True)
class Usage:
    """Token counts billed for one call (or summed over several).

    ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens`` are the headline
    counts. The breakdowns -- prompt tokens served from the provider's cache
    (``cached_prompt_tokens``), written to it (``cache_creation_tokens``), and
    hidden reasoning tokens inside ``completion_tokens`` (``reasoning_tokens``)
    -- are priced differently by most providers, so cost reporting needs them.
    Counts a provider did not report are ``0``.

    Usages add, and ``sum`` works when you give it a ``Usage()`` start (so an
    empty batch is a zero ``Usage``, not the integer ``0``):

    >>> Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15, cached_prompt_tokens=0, cache_creation_tokens=0, reasoning_tokens=0)
    >>> sum([Usage(1, 2, 3), Usage(10, 20, 30)], Usage()).total_tokens
    33
    >>> sum([], Usage()) == Usage()
    True
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_prompt_tokens: int = 0
    cache_creation_tokens: int = 0
    reasoning_tokens: int = 0

    def __add__(self, other: Any) -> "Usage":
        if isinstance(other, int) and not isinstance(other, bool) and other == 0:
            return self  # so ``sum(usages)`` (which starts at 0) works too
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            **{
                f.name: getattr(self, f.name) + getattr(other, f.name)
                for f in dataclass_fields(Usage)
            }
        )

    __radd__ = __add__

    @classmethod
    def from_response(cls, response: Any) -> Optional["Usage"]:
        """Read the ``usage`` block of a LiteLLM response; ``None`` if absent.

        >>> u = Usage.from_response({"usage": {"prompt_tokens": 3, "completion_tokens": 4}})
        >>> u.prompt_tokens, u.completion_tokens, u.total_tokens
        (3, 4, 7)
        >>> Usage.from_response({}) is None
        True
        """
        usage = _field(response, "usage")
        if usage is None:
            return None

        def first_count(paths):
            for path in paths:
                count = _as_int(_nested(usage, path))
                if count is not None:
                    return count
            return None

        counts = {name: first_count(paths) for name, paths in _USAGE_SOURCES.items()}
        if all(v is None for v in counts.values()):
            return None
        counts = {name: count or 0 for name, count in counts.items()}
        if not counts["total_tokens"]:
            counts["total_tokens"] = (
                counts["prompt_tokens"] + counts["completion_tokens"]
            )
        return cls(**counts)


@dataclass(frozen=True)
class Completion:
    """Everything a chat completion returned, not just the first text.

    Attributes:
        texts: The text of every choice, in order (``n`` of them when ``n`` was
            requested). A choice with no text content (e.g. a pure tool call)
            is ``""``; list-of-parts content is reduced to its joined text parts.
            Tool calls themselves are not surfaced here -- read them from ``raw``.
        finish_reasons: Why each choice stopped (``"stop"``, ``"length"``,
            ``"content_filter"``, ``"tool_calls"``...), aligned with ``texts``;
            ``None`` where the provider did not say. ``"length"`` means the
            text was truncated by ``max_tokens``.
        usage: Tokens billed for the whole call (all choices together), or
            ``None`` if the provider did not report usage (so filter before
            summing: ``sum((c.usage for c in cs if c.usage), Usage())``).
        model: The model id the provider reports having served the call, which
            can differ from the one requested (aliases, dated snapshots).
        raw: The untouched LiteLLM response, for anything not surfaced here.
        system_fingerprint: The provider's identifier for the serving backend
            configuration, when it reports one (OpenAI does). Seeded outputs
            are only comparable across calls with the same fingerprint. A
            response replayed from a cache carries its *original* fingerprint.
        values: With ``response_schema=``: each choice's validated value (a
            Pydantic instance for a Pydantic schema), ``None`` where invalid.
        errors: Aligned with ``values``: ``None``, or why that choice has no
            value -- the :class:`~aix.structured.SchemaValidationError` of a
            reply still invalid after the repair retries (or a refusal), or the
            exception a repair request raised.
        schema_mode: How the schema was enforced: ``"strict"``, ``"schema"``,
            ``"json_mode"`` or ``"prompt"`` (see :mod:`aix.structured`);
            ``None`` without one.
        repairs: How many repair requests were made (their tokens are in ``usage``).

    >>> c = Completion(texts=("a", "b"), finish_reasons=("stop", "length"))
    >>> c.text, c.finish_reason
    ('a', 'stop')
    """

    texts: tuple[str, ...]
    finish_reasons: tuple[Optional[str], ...] = ()
    usage: Optional[Usage] = None
    model: Optional[str] = None
    raw: Any = field(default=None, repr=False, compare=False)
    values: tuple = field(default=(), compare=False)
    errors: tuple = field(default=(), compare=False)
    schema_mode: Optional[str] = None
    repairs: int = 0
    system_fingerprint: Optional[str] = None

    @property
    def value(self) -> Any:
        """The first choice's validated value (needs ``response_schema=``).

        Raises the recorded :class:`~aix.structured.SchemaValidationError` if
        that choice still did not validate after the repair retries.
        """
        if not self.values:
            raise ValueError(
                "No structured value: call complete(..., response_schema=...)"
            )
        if self.errors and self.errors[0] is not None:
            raise self.errors[0]
        return self.values[0]

    @property
    def text(self) -> str:
        """The first choice's text (``""`` if there are no choices)."""
        return self.texts[0] if self.texts else ""

    @property
    def finish_reason(self) -> Optional[str]:
        """The first choice's finish reason."""
        return self.finish_reasons[0] if self.finish_reasons else None

    def __str__(self) -> str:
        return self.text

    @classmethod
    def from_response(cls, response: Any) -> "Completion":
        """Build a :class:`Completion` from a (non-streaming) LiteLLM response."""
        choices = _field(response, "choices") or []
        texts, reasons = [], []
        for choice in choices:
            message = _field(choice, "message")
            texts.append(_content_text(_field(message, "content")))
            reasons.append(_as_str(_field(choice, "finish_reason")))
        return cls(
            texts=tuple(texts),
            finish_reasons=tuple(reasons),
            usage=Usage.from_response(response),
            model=_as_str(_field(response, "model")),
            raw=response,
            system_fingerprint=_as_str(_field(response, "system_fingerprint")),
        )


def _require_completion():
    """Resolve litellm's ``completion`` or raise an install hint."""
    completion = _completion()
    if completion is None:
        raise ImportError(
            "LiteLLM is required for chat functionality. "
            "Install it with: pip install litellm"
        )
    return completion


def _completion_kwargs(
    prompt: Union[str, Iterable[dict]],
    *,
    model: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
    api_key: Optional[str],
) -> dict:
    """The LiteLLM ``completion`` kwargs shared by :func:`chat` and :func:`complete`.

    Normalizes the prompt, fills defaults from the active config (explicit
    arguments win), and injects the resolved API key.
    """
    messages = _normalize_prompt(prompt)

    cfg = _get_config().chat
    model = _resolve_model(model or cfg.model)
    temperature = temperature if temperature is not None else cfg.temperature
    if max_tokens is None:
        max_tokens = cfg.max_tokens

    litellm_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        litellm_kwargs["max_tokens"] = max_tokens

    # Resolve and inject the API key (explicit arg > env/.env > config store).
    # When None, fall back to LiteLLM's own implicit reading.
    resolved_key = _resolve_api_key(model, api_key=api_key)
    if resolved_key is not None:
        litellm_kwargs["api_key"] = resolved_key
    return litellm_kwargs


@_requires_credentials(lambda: _get_config().chat.model)
def chat(
    prompt: Union[str, Iterable[dict]],
    *,
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
    stream: bool = False,
    api_key: str = None,
    **kwargs,
) -> Union[str, Iterable[str]]:
    """Send a chat prompt and get a response.

    This is the main chat interface for AIX. It abstracts away provider-specific
    details and provides a clean, consistent API across all models.

    Args:
        prompt: Either a string (becomes a user message) or a list of message dicts
            with 'role' and 'content' keys
        model: Model identifier (e.g., 'gpt-4o', 'claude-sonnet-4',
            'openrouter/anthropic/claude-3.5-sonnet'). If None, uses default.
        temperature: Sampling temperature (0.0 = deterministic, 2.0 = creative).
            If None, uses default (1.0).
        max_tokens: Maximum tokens to generate. If None, uses model's default.
        stream: If True, return an iterator of text chunks. If False, return
            complete response as string.
        api_key: Explicit API key. If None, resolved from the environment / .env
            / AIX config store for the model's provider (see aix.credentials).
        **kwargs: Additional provider-specific parameters passed to LiteLLM

    Returns:
        If stream=False: Complete response as string
        If stream=True: Iterator yielding text chunks as they arrive

    Raises:
        ImportError: If LiteLLM is not installed
        ValueError: If prompt format is invalid

    Examples:
        >>> chat("What is Python?")  # doctest: +SKIP
        'Python is a high-level programming language...'

        >>> chat("Hello", model="gpt-4o")  # doctest: +SKIP
        'Hello! How can I assist you today?'

        >>> # Streaming response
        >>> for chunk in chat("Count to 5", stream=True):  # doctest: +SKIP
        ...     print(chunk, end='', flush=True)
        1, 2, 3, 4, 5

        >>> # With message history
        >>> messages = [
        ...     {"role": "system", "content": "You are a helpful assistant."},
        ...     {"role": "user", "content": "What is 2+2?"}
        ... ]
        >>> chat(messages)  # doctest: +SKIP
        '2+2 equals 4.'
    """
    completion = _require_completion()
    litellm_kwargs = _completion_kwargs(
        prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
    )
    litellm_kwargs["stream"] = stream

    # Add any additional provider-specific kwargs
    litellm_kwargs.update(kwargs)

    # Call LiteLLM
    response = completion(**litellm_kwargs)

    # Extract and return text
    if stream:
        return _extract_text_from_stream(response)
    else:
        return _extract_text_from_response(response)


def _check_max_repairs(max_repairs: Any) -> int:
    """``int(max_repairs)`` if it is a non-negative integer, else ``ValueError``."""
    if (
        isinstance(max_repairs, bool)
        or not isinstance(max_repairs, numbers.Integral)
        or max_repairs < 0
    ):
        raise ValueError(
            f"max_repairs must be a non-negative integer, got {max_repairs!r}"
        )
    return int(max_repairs)


def _validate_complete_args(func):
    """Reject bad :func:`complete` arguments before the credential preflight.

    Kept outside :func:`complete` (and outermost in its decorator stack) so a
    caller error such as ``n=0`` surfaces as itself, not as a missing API key.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        n = kwargs.get("n", 1)
        if not isinstance(n, numbers.Integral) or isinstance(n, bool) or n < 1:
            raise ValueError(f"n must be a positive integer, got {n!r}")
        if kwargs.get("stream"):
            raise ValueError(
                "complete() does not stream; use chat(..., stream=True) for streaming."
            )
        if (
            kwargs.get("response_schema") is not None
            and kwargs.get("response_format") is not None
        ):
            raise ValueError("Pass response_schema or response_format, not both.")
        if kwargs.get("max_repairs") is not None:
            _check_max_repairs(kwargs["max_repairs"])
        return func(*args, **kwargs)

    return wrapper


@_validate_complete_args
@_requires_credentials(lambda: _get_config().chat.model)
def complete(
    prompt: Union[str, Iterable[dict]],
    *,
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
    n: int = 1,
    seed: int = None,
    response_format: Union[dict, type] = None,
    response_schema: Union[dict, type] = None,
    max_repairs: int = None,
    api_key: str = None,
    **kwargs,
) -> Completion:
    """Send a chat prompt and get the whole response back as a :class:`Completion`.

    The non-lossy sibling of :func:`chat`: same inputs and defaults, but instead
    of the first choice's text it returns every choice (``texts``), why each one
    stopped (``finish_reasons``), the tokens billed (``usage``), the model that
    served the call, and the raw response.

    Args:
        prompt: A string (becomes a user message) or a list of message dicts.
        model: Model identifier or alias. If None, uses the configured default.
        temperature: Sampling temperature. If None, uses the configured default.
        max_tokens: Maximum tokens per choice. If None, uses the configured default.
        n: How many choices to ask the provider for in this one call. Only
            sent when ``n != 1``; providers without native ``n`` support reject
            it (see LiteLLM's ``get_supported_openai_params``).
        seed: Best-effort reproducibility seed. Only sent when given. Support is
            provider-dependent (``aix.supports(model, "seed")``); where it is
            unsupported LiteLLM either drops it or rejects the call, and where
            it is supported determinism is still best-effort -- compare
            ``system_fingerprint`` across calls before reading a difference
            as model instability.
        response_format: Passed through to LiteLLM (e.g. ``{"type": "json_object"}``).
            Mutually exclusive with ``response_schema``.
        response_schema: A JSON Schema dict or a Pydantic model class. The
            provider is asked to enforce it as strongly as the model allows
            (strict ``json_schema`` > non-strict ``json_schema`` > JSON mode plus
            the schema in the prompt; see :mod:`aix.structured`), and every
            choice is validated client-side into ``values`` / ``errors``.
        max_repairs: With ``response_schema``: how many times to re-ask for an
            invalid choice, showing the model its error (``None`` = config
            ``chat.schema_max_repairs``; ``0`` disables repair).
        api_key: Explicit API key; otherwise resolved as for :func:`chat`.
        **kwargs: Additional provider-specific parameters passed to LiteLLM.

    Returns:
        A :class:`Completion`.

    Raises:
        ImportError: If LiteLLM is not installed.
        ValueError: If ``n`` is not a positive integer, or the prompt is invalid.

    Examples:
        >>> c = complete("Name a color", n=3)  # doctest: +SKIP
        >>> c.texts, c.finish_reasons  # doctest: +SKIP
        (('Blue', 'Green', 'Red'), ('stop', 'stop', 'stop'))
        >>> c.usage  # doctest: +SKIP
        Usage(prompt_tokens=11, completion_tokens=6, total_tokens=17)
        >>> complete("Write an essay", max_tokens=5).finish_reason  # doctest: +SKIP
        'length'

        Structured output, validated:

        >>> from pydantic import BaseModel  # doctest: +SKIP
        >>> class Verdict(BaseModel):  # doctest: +SKIP
        ...     label: str
        ...     confidence: float
        >>> c = complete("Is the sky green?", response_schema=Verdict)  # doctest: +SKIP
        >>> c.value, c.schema_mode  # doctest: +SKIP
        (Verdict(label='no', confidence=0.99), 'strict')
    """
    n = int(n)  # already validated (see _validate_complete_args); np ints too
    completion = _require_completion()
    litellm_kwargs = _completion_kwargs(
        prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
    )
    optional = {
        "n": n if n != 1 else None,
        "seed": seed,
        "response_format": response_format,
    }
    litellm_kwargs.update({k: v for k, v in optional.items() if v is not None})

    if response_schema is None:
        litellm_kwargs.update(kwargs)
        return Completion.from_response(completion(**litellm_kwargs))

    if max_repairs is None:
        max_repairs = _check_max_repairs(_get_config().chat.schema_max_repairs)
    spec = _ResponseSchema.from_schema(response_schema)
    mode = spec.mode_for(litellm_kwargs["model"])
    response_format = spec.response_format(mode)
    if response_format is not None:
        litellm_kwargs["response_format"] = response_format
    if mode in _SCHEMA_IN_PROMPT_MODES:
        litellm_kwargs["messages"] = _with_instruction(
            litellm_kwargs["messages"], spec.instruction()
        )
    litellm_kwargs.update(kwargs)
    return _validated(
        completion,
        Completion.from_response(completion(**litellm_kwargs)),
        litellm_kwargs,
        spec=spec,
        mode=mode,
        max_repairs=max_repairs,
    )


def _with_instruction(messages: list, instruction: str) -> list:
    """``messages`` with ``instruction`` added to the final user turn (copied)."""
    messages = [dict(m) for m in messages]
    last = messages[-1] if messages else None
    if last and last.get("role") == "user" and isinstance(last.get("content"), str):
        last["content"] = f"{last['content']}\n\n{instruction}"
    else:
        messages.append({"role": "user", "content": instruction})
    return messages


def _refusals(response: Any, n_choices: int) -> list:
    """Each choice's ``message.refusal`` text (OpenAI structured outputs), or ``None``."""
    choices = _field(response, "choices") or []
    found = [_as_str(_field(_field(c, "message"), "refusal")) for c in choices]
    return (found + [None] * n_choices)[:n_choices]


def _validated(
    completion_fn,
    first: Completion,
    litellm_kwargs: dict,
    *,
    spec,
    mode: str,
    max_repairs: int,
) -> Completion:
    """Validate every choice of ``first``, repairing invalid ones up to ``max_repairs``.

    A repair is one extra ``n=1`` request: the conversation, the invalid reply
    as the assistant turn, and the validation error as the next user turn. Its
    tokens are added to ``usage``; the repaired text replaces the choice's.
    """
    from aix.structured import SchemaValidationError

    texts, reasons = list(first.texts), list(first.finish_reasons)
    refusals = _refusals(first.raw, len(texts))
    values, errors = [], []
    usages = [first.usage] if first.usage is not None else []
    repairs = 0
    repair_kwargs = {k: v for k, v in litellm_kwargs.items() if k != "n"}

    def validate_choice(i: int, text: str):
        """``(value, error)`` for choice ``i``, repairing it in place if needed."""
        nonlocal repairs
        if refusals[i]:
            # A refusal is an answer, not a formatting slip: never "repair" it.
            return None, SchemaValidationError(
                f"The model refused: {refusals[i]}", text=text, detail="refused"
            )
        for attempt in range(max_repairs + 1):
            try:
                return spec.validate(text), None
            except SchemaValidationError as e:
                if attempt == max_repairs:
                    return None, e
                repair_kwargs["messages"] = [
                    *litellm_kwargs["messages"],
                    {"role": "assistant", "content": text or _EMPTY_REPLY_PLACEHOLDER},
                    {
                        "role": "user",
                        "content": f"That reply was invalid ({e.detail}). "
                        + spec.instruction(),
                    },
                ]
                try:
                    fixed = Completion.from_response(completion_fn(**repair_kwargs))
                except Exception as request_error:
                    # Keep the other choices: record this one as failed.
                    return None, request_error
                finally:
                    repairs += 1
                if fixed.usage is not None:
                    usages.append(fixed.usage)
                refusal = _refusals(fixed.raw, 1)[0]
                if refusal:
                    texts[i] = fixed.text
                    return None, SchemaValidationError(
                        f"The model refused: {refusal}",
                        text=fixed.text,
                        detail="refused",
                    )
                text = texts[i] = fixed.text
                if i < len(reasons):
                    reasons[i] = fixed.finish_reason
        return None, None  # pragma: no cover - loop always returns

    for i, text in enumerate(texts):
        value, error = validate_choice(i, text)
        values.append(value)
        errors.append(error)

    return _replace_dataclass(
        first,
        texts=tuple(texts),
        finish_reasons=tuple(reasons),
        usage=sum(usages, Usage()) if usages else None,
        values=tuple(values),
        errors=tuple(errors),
        schema_mode=mode,
        repairs=repairs,
    )


def chat_with_history(
    system_prompt: str = None, *, model: str = None, **chat_kwargs
) -> "ChatSession":
    """Create a stateful chat session that maintains conversation history.

    Args:
        system_prompt: Optional system message to set context/behavior
        model: Model to use for this session
        **chat_kwargs: Additional parameters passed to chat()

    Returns:
        ChatSession object with send() method

    Examples:
        >>> session = chat_with_history("You are a helpful math tutor")  # doctest: +SKIP
        >>> session.send("What is 2+2?")  # doctest: +SKIP
        'The answer is 4.'
        >>> session.send("And if I add 3 to that?")  # doctest: +SKIP
        'That would be 7.'
        >>> len(session.history)  # doctest: +SKIP
        5
    """
    return ChatSession(system_prompt=system_prompt, model=model, **chat_kwargs)


class ChatSession:
    """Stateful chat session that maintains conversation history.

    This class provides a convenient way to have multi-turn conversations
    without manually managing message history.

    Examples:
        >>> session = ChatSession()  # doctest: +SKIP
        >>> response = session.send("My name is Alice")  # doctest: +SKIP
        >>> response = session.send("What's my name?")  # doctest: +SKIP
        'Your name is Alice.'
    """

    def __init__(self, system_prompt: str = None, *, model: str = None, **chat_kwargs):
        """Initialize a new chat session.

        Args:
            system_prompt: Optional system message
            model: Model to use for this session
            **chat_kwargs: Additional parameters for chat()
        """
        self.model = model
        self.chat_kwargs = chat_kwargs
        self.history = []

        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

    def send(self, message: str, **kwargs) -> str:
        """Send a message and get a response.

        Args:
            message: User message to send
            **kwargs: Override chat parameters for this message

        Returns:
            Assistant's response
        """
        # Add user message to history
        self.history.append({"role": "user", "content": message})

        # Merge kwargs
        call_kwargs = {**self.chat_kwargs, **kwargs}

        # Get response
        response = chat(self.history, model=self.model, **call_kwargs)

        # Add assistant response to history
        self.history.append({"role": "assistant", "content": response})

        return response

    def clear_history(self, keep_system: bool = True):
        """Clear conversation history.

        Args:
            keep_system: If True, preserve system message (if any)
        """
        if keep_system and self.history and self.history[0]["role"] == "system":
            self.history = [self.history[0]]
        else:
            self.history = []


# Convenience function for simple question-answering
def ask(question: str, model: str = None, **kwargs) -> str:
    """Ask a single question and get an answer.

    This is a convenience wrapper around chat() for simple Q&A.

    Args:
        question: The question to ask
        model: Model to use
        **kwargs: Additional parameters for chat()

    Returns:
        The answer as a string

    Examples:
        >>> from aix.chat import ask
        >>> ask("What is the capital of France?")  # doctest: +SKIP
        'The capital of France is Paris.'
    """
    return chat(question, model=model, **kwargs)
