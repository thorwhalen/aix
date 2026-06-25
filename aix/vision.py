"""Image-to-text (vision) interface for AIX.

The cross-modal counterpart to :mod:`aix.image` (which is text→image): given an
image, produce text — a caption, an answer to a question about it, or a
structured judgement. Like the rest of AIX it is a thin, provider-neutral facade
over LiteLLM's multimodal ``completion`` (the same backend ``chat`` uses), so the
same call routes to any vision-capable provider (OpenAI, Anthropic, Gemini,
OpenRouter, …) by model id alone.

:func:`describe_image` is the primitive: image (URL / local path / bytes / PIL
image / ``data:`` URI) + a prompt → text. :func:`to_image_content` exposes the
multimodal content block builder so callers can assemble richer messages (e.g.
multiple images in one turn) and pass them straight to :func:`aix.chat`.

Examples:
    Caption an image (any vision-capable model):

    >>> from aix.vision import describe_image
    >>> describe_image("https://example.com/cat.jpg")  # doctest: +SKIP
    'A grey tabby cat sitting on a windowsill in afternoon light.'

    Ask a specific question:

    >>> describe_image("photo.png", prompt="What colour is the car?")  # doctest: +SKIP
    'The car is red.'

    Build a multi-image message yourself and hand it to ``chat``:

    >>> from aix import chat
    >>> from aix.vision import to_image_content
    >>> msg = [{"role": "user", "content": [          # doctest: +SKIP
    ...     {"type": "text", "text": "Which is brighter?"},
    ...     to_image_content("a.jpg"), to_image_content("b.jpg"),
    ... ]}]
    >>> chat(msg, model="gpt-4o")  # doctest: +SKIP
    'The second image is brighter.'

:func:`compare_images` builds on these primitives: it puts a *candidate* image
beside one or more *reference* images and asks a vision model, via a structured
JSON path, for an explainable per-aspect likeness verdict — the "explain the
drift" half of a reference supervisor (face / costume / setting / lighting /
props matched?). The numeric identity-cosine gate lives elsewhere (lookbook);
this layer is the explainable checklist on top of it.

    >>> from aix.vision import compare_images
    >>> verdict = compare_images("gen.png", "locked_ref.png")  # doctest: +SKIP
    >>> verdict.match, verdict["identity"].match  # doctest: +SKIP
    (True, True)
"""

from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence, Union

from aix.config import (
    get_config as _get_config,
    resolve_model as _resolve_model,
    VisionConfig as _VisionConfig,
)
from aix.credentials import (
    resolve_api_key as _resolve_api_key,
    requires_credentials as _requires_credentials,
)

# LiteLLM is a soft dependency, imported privately (mirrors aix.chat).
try:
    from litellm import completion as _litellm_completion
except ImportError:  # pragma: no cover - exercised only without litellm
    _litellm_completion = None

__all__ = [
    "describe_image",
    "to_image_content",
    "compare_images",
    "RubricVerdict",
    "ImageComparison",
    "DFLT_VISION_MODEL",
    "DFLT_VISION_PROMPT",
    "DFLT_COMPARE_RUBRIC",
    "DFLT_FILM_RUBRIC",
]

#: Shipped-default vision model, kept for reference. The *active* default is
#: resolved from ``aix.config`` at call time (see :class:`aix.config.VisionConfig`).
DFLT_VISION_MODEL = _VisionConfig().model

#: Default instruction when the caller doesn't supply one.
DFLT_VISION_PROMPT = "Describe this image in detail."

#: Default rubric for :func:`compare_images` — a generic likeness checklist.
#: Each entry is one aspect the vision model judges independently.
DFLT_COMPARE_RUBRIC = ("identity", "costume", "setting", "lighting", "props")

#: Filmmaking-oriented rubric (Noel's reference-supervisor checklist): the
#: locked face, the architecture/set, the lighting, and the props. Pass it as
#: ``rubric=`` when comparing a generated frame against a locked reference.
DFLT_FILM_RUBRIC = (
    "face_identity",
    "costume",
    "setting_architecture",
    "lighting",
    "props",
    "skin_realism",
)

#: Default JPEG when raw bytes can't be sniffed (most stock images are JPEG).
_DFLT_IMAGE_MIME = "image/jpeg"

#: Magic-number prefixes for sniffing a MIME type from raw bytes.
_MAGIC_MIME = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"RIFF", "image/webp"),  # RIFF....WEBP; prefix is enough to distinguish here
    (b"BM", "image/bmp"),
)


def to_image_content(
    image: "Union[str, Path, bytes, Any]", *, detail: "str | None" = None
) -> dict:
    """Build a multimodal ``image_url`` content block for ``image``.

    ``image`` may be:

    - an ``http(s)://`` URL or a ``data:`` URI — passed through verbatim;
    - a local file path (``str`` or :class:`~pathlib.Path`) — read and inlined
      as a base64 ``data:`` URI with a guessed MIME type;
    - raw ``bytes`` — base64-inlined (MIME sniffed from magic bytes, else JPEG);
    - a PIL ``Image`` — encoded to PNG and inlined.

    ``detail`` (``"low"`` | ``"high"`` | ``"auto"``) is forwarded when set; the
    block is the OpenAI/LiteLLM multimodal shape understood across providers.

    >>> to_image_content("https://x/y.jpg")
    {'type': 'image_url', 'image_url': {'url': 'https://x/y.jpg'}}
    >>> to_image_content("data:image/png;base64,AAAA")["image_url"]["url"][:10]
    'data:image'
    """
    url = _to_image_url(image)
    image_url: dict[str, Any] = {"url": url}
    if detail is not None:
        image_url["detail"] = detail
    return {"type": "image_url", "image_url": image_url}


@_requires_credentials(lambda: _get_config().vision.model)
def describe_image(
    image: "Union[str, Path, bytes, Any]",
    *,
    prompt: str = DFLT_VISION_PROMPT,
    model: "str | None" = None,
    api_key: "str | None" = None,
    max_tokens: "int | None" = None,
    temperature: "float | None" = None,
    detail: "str | None" = None,
    **kwargs: Any,
) -> str:
    """Describe (or answer a question about) ``image`` and return the text.

    The image→text primitive. ``image`` accepts a URL, a local path, raw bytes,
    a PIL image, or a ``data:`` URI (see :func:`to_image_content`). ``prompt`` is
    the instruction (default: a generic "describe this image"); pass a question
    for VQA or a rubric for a judgement. Everything past ``image`` is keyword.

    Args:
        image: The image (URL / path / bytes / PIL image / ``data:`` URI).
        prompt: The text instruction accompanying the image.
        model: Vision-capable model id (e.g. ``gpt-4o``, ``claude-sonnet-4-6``,
            ``gemini/gemini-1.5-pro``) or an alias. ``None`` → the configured
            default (:class:`aix.config.VisionConfig`).
        api_key: Explicit API key; ``None`` resolves it from the environment /
            ``.env`` / config store for the model's provider.
        max_tokens: Cap on generated tokens (``None`` → provider default).
        temperature: Sampling temperature (``None`` → provider default).
        detail: Image detail hint (``"low"`` | ``"high"`` | ``"auto"``).
        **kwargs: Extra provider-specific params forwarded to LiteLLM.

    Returns:
        The model's text response.

    Raises:
        ImportError: If LiteLLM is not installed.

    >>> describe_image("cat.jpg", prompt="Caption it.")  # doctest: +SKIP
    'A cat on a sofa.'
    """
    if _litellm_completion is None:
        raise ImportError(
            "LiteLLM is required for vision functionality. "
            "Install it with: pip install litellm"
        )

    cfg = _get_config().vision
    model = _resolve_model(model or cfg.model)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                to_image_content(image, detail=detail),
            ],
        }
    ]

    litellm_kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if max_tokens is not None:
        litellm_kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        litellm_kwargs["temperature"] = temperature

    resolved_key = _resolve_api_key(model, api_key=api_key)
    if resolved_key is not None:
        litellm_kwargs["api_key"] = resolved_key

    litellm_kwargs.update(kwargs)

    response = _litellm_completion(**litellm_kwargs)
    return _extract_text(response)


# --- compare_images: structured two-image likeness check --------------------


@dataclass(frozen=True)
class RubricVerdict:
    """A vision model's verdict on a single rubric aspect.

    Attributes:
        aspect: The rubric item this verdict is about (e.g. ``"identity"``).
        match: Whether the candidate matches the reference for this aspect.
        confidence: The model's self-reported confidence in ``[0.0, 1.0]``.
        note: A short free-text explanation of the verdict.
    """

    aspect: str
    match: bool
    confidence: float
    note: str = ""


@dataclass(frozen=True)
class ImageComparison:
    """Structured result of comparing a candidate image to reference(s).

    Behaves like an ordered, read-only mapping of ``aspect -> RubricVerdict``
    (``comparison["identity"]``, ``in``, iteration, ``len``) so per-aspect
    lookups read naturally, while also carrying the overall verdict.

    Attributes:
        match: Overall pass/fail across the whole rubric.
        confidence: Overall confidence in ``[0.0, 1.0]``.
        explanation: A short overall summary of the comparison.
        aspects: The per-aspect verdicts, one :class:`RubricVerdict` per
            rubric item, in rubric order.
        model: The vision model id that produced the verdict.
    """

    match: bool
    confidence: float
    explanation: str
    aspects: tuple[RubricVerdict, ...] = field(default_factory=tuple)
    model: "str | None" = None

    def _by_aspect(self) -> "dict[str, RubricVerdict]":
        return {verdict.aspect: verdict for verdict in self.aspects}

    def __getitem__(self, aspect: str) -> RubricVerdict:
        return self._by_aspect()[aspect]

    def __contains__(self, aspect: object) -> bool:
        return aspect in self._by_aspect()

    def __iter__(self):
        return iter(self._by_aspect())

    def __len__(self) -> int:
        return len(self.aspects)

    def get(
        self, aspect: str, default: "RubricVerdict | None" = None
    ) -> "RubricVerdict | None":
        """Return the verdict for ``aspect``, or ``default`` if absent."""
        return self._by_aspect().get(aspect, default)


#: Instruction prefix for the comparison prompt. Kept here (not inlined) so the
#: phrasing is a single, tunable source of truth.
_COMPARE_INSTRUCTION = (
    "You are a strict visual continuity supervisor. The FIRST image is the "
    "CANDIDATE; the remaining image(s) are the locked REFERENCE the candidate "
    "must match. Judge each rubric aspect INDEPENDENTLY (do not let one aspect "
    "color another). For each aspect decide whether the candidate matches the "
    "reference, give a confidence in [0.0, 1.0], and a short note explaining "
    "what matches or drifts. Then give an overall match (true only if every "
    "important aspect matches), an overall confidence, and a one-sentence "
    "explanation."
)


@_requires_credentials(lambda: _get_config().vision.model)
def compare_images(
    candidate: "Union[str, Path, bytes, Any]",
    reference: "Union[str, Path, bytes, Any, Sequence[Any]]",
    *,
    rubric: "Sequence[str]" = DFLT_COMPARE_RUBRIC,
    model: "str | None" = None,
    api_key: "str | None" = None,
    max_tokens: "int | None" = None,
    temperature: "float | None" = 0.0,
    detail: "str | None" = None,
    instruction: str = _COMPARE_INSTRUCTION,
    **kwargs: Any,
) -> ImageComparison:
    """Compare a ``candidate`` image to ``reference`` image(s) on a rubric.

    The explainable half of a reference supervisor: a vision model returns a
    per-aspect pass/fail checklist (does the face match? the costume? the set?
    the lighting? the props?) plus an overall verdict — the explainable layer
    over a cheap numeric identity-cosine gate (which lives elsewhere, e.g.
    lookbook). Built on :func:`to_image_content` (multi-image content block)
    and the same multimodal ``completion`` path as :func:`describe_image`,
    asked via a JSON contract for a structured answer.

    The comparison is *pointwise* (each aspect judged on its own), not a ranked
    pairwise comparison, to avoid position bias.

    Args:
        candidate: The image under review (URL / path / bytes / PIL image /
            ``data:`` URI — anything :func:`to_image_content` accepts).
        reference: The locked reference — a single image *or* a sequence of
            images (a locked set) in the same flexible formats. An empty
            sequence is an error.
        rubric: The aspects to evaluate, one verdict per item. Defaults to
            :data:`DFLT_COMPARE_RUBRIC`; pass :data:`DFLT_FILM_RUBRIC` (or any
            custom sequence) to override — e.g. ``("face", "architecture",
            "props", "lighting")``. Must be non-empty.
        model: Vision-capable model id or alias. ``None`` → the configured
            default (:class:`aix.config.VisionConfig`).
        api_key: Explicit API key; ``None`` resolves it for the model's
            provider from the environment / ``.env`` / config store.
        max_tokens: Cap on generated tokens (``None`` → provider default).
        temperature: Sampling temperature (default ``0.0`` for a stable,
            reproducible verdict; ``None`` → provider default).
        detail: Image detail hint (``"low"`` | ``"high"`` | ``"auto"``),
            applied to every image block.
        instruction: The system-style framing prepended to the rubric. Has a
            sensible default; override to retune the supervisor's strictness.
        **kwargs: Extra provider-specific params forwarded to LiteLLM.

    Returns:
        An :class:`ImageComparison` — overall ``match`` / ``confidence`` /
        ``explanation`` plus an ordered, mapping-like collection of
        :class:`RubricVerdict` (one per rubric aspect, keyed by aspect).

    Raises:
        ImportError: If LiteLLM is not installed.
        ValueError: If ``rubric`` is empty or ``reference`` is an empty
            sequence, or if the model's reply can't be parsed as the expected
            JSON verdict.

    Examples:
        Default rubric, single reference:

        >>> compare_images("gen.png", "ref.png")  # doctest: +SKIP
        ImageComparison(match=True, confidence=0.9, ...)

        Filmmaking rubric, a locked reference *set*:

        >>> from aix.vision import DFLT_FILM_RUBRIC
        >>> compare_images(  # doctest: +SKIP
        ...     candidate="frame_042.png",
        ...     reference=["ref_front.png", "ref_side.png"],
        ...     rubric=DFLT_FILM_RUBRIC,
        ... )["face_identity"].match
        True
    """
    # Validate arguments before checking the optional dependency, so a bad call
    # (empty rubric / empty reference set) is reported the same way regardless of
    # whether LiteLLM happens to be installed.
    rubric = tuple(rubric)
    if not rubric:
        raise ValueError("`rubric` must contain at least one aspect to evaluate.")

    references = _as_image_sequence(reference)
    if not references:
        raise ValueError(
            "`reference` must be at least one image (got an empty sequence)."
        )

    if _litellm_completion is None:
        raise ImportError(
            "LiteLLM is required for vision functionality. "
            "Install it with: pip install litellm"
        )

    cfg = _get_config().vision
    model = _resolve_model(model or cfg.model)

    text_block = {"type": "text", "text": _compare_prompt(instruction, rubric)}
    image_blocks = [to_image_content(candidate, detail=detail)] + [
        to_image_content(ref, detail=detail) for ref in references
    ]
    messages = [{"role": "user", "content": [text_block, *image_blocks]}]

    litellm_kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if max_tokens is not None:
        litellm_kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        litellm_kwargs["temperature"] = temperature

    resolved_key = _resolve_api_key(model, api_key=api_key)
    if resolved_key is not None:
        litellm_kwargs["api_key"] = resolved_key

    litellm_kwargs.update(kwargs)

    response = _litellm_completion(**litellm_kwargs)
    raw = _extract_text(response)
    return _parse_comparison(raw, rubric=rubric, model=model)


# --- internals --------------------------------------------------------------


def _as_image_sequence(reference: Any) -> "list[Any]":
    """Normalise ``reference`` to a list of image inputs.

    A single image (URL str / path / bytes / PIL image / ``data:`` URI) becomes
    a one-element list; a non-string, non-bytes sequence is taken as a locked
    set and returned as a list. ``str`` and ``bytes`` are *atomic* images here,
    never iterated character/byte-wise.
    """
    if isinstance(reference, (str, bytes, Path)):
        return [reference]
    if hasattr(reference, "save"):  # a PIL image is iterable-ish; treat as atomic
        return [reference]
    if isinstance(reference, Sequence):
        return list(reference)
    # Any other iterable (generator, tuple already handled by Sequence) — be lenient.
    try:
        return list(reference)
    except TypeError:
        return [reference]


def _compare_prompt(instruction: str, rubric: "tuple[str, ...]") -> str:
    """Build the JSON-contract prompt asking for a per-aspect verdict."""
    aspect_list = ", ".join(rubric)
    schema = {
        "match": "boolean — overall match across the whole rubric",
        "confidence": "number in [0.0, 1.0] — overall confidence",
        "explanation": "string — one-sentence overall summary",
        "aspects": [
            {
                "aspect": "string — one of the rubric aspects, verbatim",
                "match": "boolean",
                "confidence": "number in [0.0, 1.0]",
                "note": "string — short explanation",
            }
        ],
    }
    return (
        f"{instruction}\n\n"
        f"Rubric aspects to evaluate (one verdict each, use these exact names): "
        f"{aspect_list}.\n\n"
        f"Respond with ONLY valid JSON of this shape (no prose, no code fence):\n"
        f"{json.dumps(schema)}\n"
        f"Include exactly one entry in \"aspects\" for each rubric aspect above."
    )


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "match", "pass"}
    return bool(value)


def _coerce_confidence(value: Any) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, conf))


def _parse_comparison(
    raw: str, *, rubric: "tuple[str, ...]", model: "str | None"
) -> ImageComparison:
    """Parse the model's JSON reply into an :class:`ImageComparison`.

    Tolerant of a markdown code fence around the JSON; raises ``ValueError`` if
    no JSON object can be recovered. Missing aspects are filled with a
    low-confidence non-match so the result always covers the full rubric.
    """
    data = _loads_lenient(raw)
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object verdict from the model, got: {raw[:120]!r}"
        )

    by_aspect: dict[str, RubricVerdict] = {}
    for item in data.get("aspects", []) or []:
        if not isinstance(item, dict):
            continue
        aspect = str(item.get("aspect", "")).strip()
        if not aspect:
            continue
        by_aspect[aspect] = RubricVerdict(
            aspect=aspect,
            match=_coerce_bool(item.get("match")),
            confidence=_coerce_confidence(item.get("confidence")),
            note=str(item.get("note", "")),
        )

    # Emit one verdict per requested rubric aspect, in rubric order, filling any
    # the model omitted with a conservative non-match.
    ordered = tuple(
        by_aspect.get(
            aspect,
            RubricVerdict(
                aspect=aspect,
                match=False,
                confidence=0.0,
                note="No verdict returned for this aspect.",
            ),
        )
        for aspect in rubric
    )

    return ImageComparison(
        match=_coerce_bool(data.get("match")),
        confidence=_coerce_confidence(data.get("confidence")),
        explanation=str(data.get("explanation", "")),
        aspects=ordered,
        model=model,
    )


def _loads_lenient(text: str) -> Any:
    """Parse JSON, tolerating a surrounding ```/```json code fence or prose.

    Tries a direct parse, then a fence strip, then the first ``{...}`` span.
    Raises ``ValueError`` if nothing parses.
    """
    stripped = text.strip()
    candidates = [stripped]
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        inner = "\n".join(lines[1:-1])
        if inner.startswith("json"):
            inner = inner[4:].strip()
        candidates.append(inner)
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(stripped[start : end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
    raise ValueError(
        f"Could not parse a JSON verdict from the model reply: {text[:120]!r}"
    )


def _to_image_url(image: "Union[str, Path, bytes, Any]") -> str:
    """Resolve ``image`` to a URL or base64 ``data:`` URI for a content block."""
    if isinstance(image, Path):
        return _data_uri(*_read_path(image))
    if isinstance(image, bytes):
        return _data_uri(image, _sniff_mime(image))
    if isinstance(image, str):
        if image.startswith(("http://", "https://", "data:")):
            return image
        return _data_uri(*_read_path(Path(image)))
    # Treat anything else as a PIL image (duck-typed: has .save).
    if hasattr(image, "save"):
        return _data_uri(_pil_to_png_bytes(image), "image/png")
    raise TypeError(
        f"Unsupported image type {type(image)!r}; pass a URL, path, bytes, "
        "PIL image, or data: URI."
    )


def _read_path(path: Path) -> "tuple[bytes, str]":
    data = path.read_bytes()
    mime = mimetypes.guess_type(str(path))[0] or _sniff_mime(data)
    return data, mime


def _sniff_mime(data: bytes) -> str:
    for prefix, mime in _MAGIC_MIME:
        if data.startswith(prefix):
            return mime
    return _DFLT_IMAGE_MIME


def _data_uri(data: bytes, mime: str) -> str:
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _pil_to_png_bytes(image: Any) -> bytes:
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _extract_text(response: Any) -> str:
    """Pull text out of a LiteLLM response (mirrors aix.chat)."""
    try:
        return response.choices[0].message.content or ""
    except (AttributeError, IndexError, KeyError):  # pragma: no cover - defensive
        return str(response)
