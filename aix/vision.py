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
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Union

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
    "DFLT_VISION_MODEL",
    "DFLT_VISION_PROMPT",
]

#: Shipped-default vision model, kept for reference. The *active* default is
#: resolved from ``aix.config`` at call time (see :class:`aix.config.VisionConfig`).
DFLT_VISION_MODEL = _VisionConfig().model

#: Default instruction when the caller doesn't supply one.
DFLT_VISION_PROMPT = "Describe this image in detail."

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


# --- internals --------------------------------------------------------------


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
