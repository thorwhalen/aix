"""Central configuration for AIX (single source of truth for defaults).

This module is the SSOT for *what model (and related parameters) each AIX
function uses by default*. Instead of scattering hardcoded ``DFLT_*`` constants
across modules, every function resolves its defaults from a single, layered
configuration.

Resolution layers, highest precedence first:

1. **Explicit call argument** -- e.g. ``chat(..., model=...)`` always wins
   (handled at the call site, not here).
2. **Runtime override** -- :func:`configure` swaps the active config; :func:`using`
   scopes an override to a ``with`` block.
3. **Environment variables** -- ``AIX_CHAT_MODEL``, ``AIX_CHAT_TEMPERATURE``,
   ``AIX_EMBEDDING_MODEL``, ``AIX_IMAGE_MODEL``, ``AIX_TTS_MODEL``,
   ``AIX_TTS_VOICE``, ``AIX_TRANSCRIPTION_MODEL``, ...
4. **User config file** -- a TOML file at ``<app config dir>/config.toml``
   (override the path with the ``AIX_CONFIG_FILE`` environment variable).
5. **Shipped defaults** -- the dataclass field defaults below.

The schema is a set of frozen dataclasses (immutable). A module-level *active
config* pointer holds the currently resolved :class:`AixConfig`; :func:`configure`
and :func:`using` swap that pointer rather than mutating in place.

Examples:
    >>> from aix import config
    >>> c = config.get_config()
    >>> isinstance(c.chat.model, str)
    True

    Scoped override (does not leak outside the block):

    >>> with config.using(chat_model="some/model"):
    ...     config.get_config().chat.model
    'some/model'
    >>> config.get_config().chat.model == c.chat.model
    True

    TOML config file shape::

        [chat]
        model = "openai/gpt-4o-mini"
        temperature = 1.0

        [embeddings]
        model = "text-embedding-3-small"

        [image]
        model = "dall-e-3"
        size = "1024x1024"

        [audio]
        tts_model = "tts-1"
        tts_voice = "alloy"
        transcription_model = "whisper-1"

        [aliases]
        fast = "openai/gpt-4o-mini"
        best = "anthropic/claude-sonnet-4"
"""

import os
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, replace, is_dataclass, MISSING
from typing import Any, Callable, Optional, Mapping, Iterator

try:  # Python 3.11+
    import tomllib as _tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for older interpreters
    try:
        import tomli as _tomllib  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover
        _tomllib = None


__all__ = [
    "ChatConfig",
    "EmbeddingConfig",
    "ImageConfig",
    "AudioConfig",
    "VideoConfig",
    "AixConfig",
    "load_config",
    "get_config",
    "set_config",
    "configure",
    "using",
    "config_file_path",
]


def _opt(default: Any, *, flat: str, cast: Callable[[str], Any] = str):
    """Define a config field.

    Args:
        default: The shipped default value.
        flat: The flat override name (used both for ``configure(**)`` kwargs and,
            uppercased and ``AIX_``-prefixed, for the environment variable).
            e.g. ``flat="chat_model"`` -> env var ``AIX_CHAT_MODEL``.
        cast: Callable used to coerce a string (from env var) to the field type.
    """
    return field(default=default, metadata={"flat": flat, "cast": cast})


def _env_name(f) -> str:
    """Environment-variable name for a config field (``AIX_`` + flat, uppercased)."""
    return "AIX_" + f.metadata["flat"].upper()


@dataclass(frozen=True)
class ChatConfig:
    """Defaults for ``chat`` / ``ask`` / ``prompt_func``."""

    model: str = _opt("gpt-4o-mini", flat="chat_model")
    temperature: float = _opt(1.0, flat="chat_temperature", cast=float)
    max_tokens: Optional[int] = _opt(None, flat="chat_max_tokens", cast=int)


@dataclass(frozen=True)
class EmbeddingConfig:
    """Defaults for ``embeddings`` / ``embed``."""

    model: str = _opt("text-embedding-3-small", flat="embedding_model")
    batch_size: int = _opt(512, flat="embedding_batch_size", cast=int)


@dataclass(frozen=True)
class ImageConfig:
    """Defaults for ``generate_image`` / ``generate_images``."""

    model: str = _opt("dall-e-2", flat="image_model")
    size: str = _opt("1024x1024", flat="image_size")
    quality: str = _opt("standard", flat="image_quality")
    num_images: int = _opt(1, flat="image_num_images", cast=int)


@dataclass(frozen=True)
class AudioConfig:
    """Defaults for ``text_to_speech`` / ``transcribe`` / ``translate_audio``."""

    tts_model: str = _opt("tts-1", flat="tts_model")
    tts_voice: str = _opt("alloy", flat="tts_voice")
    tts_speed: float = _opt(1.0, flat="tts_speed", cast=float)
    transcription_model: str = _opt("whisper-1", flat="transcription_model")


@dataclass(frozen=True)
class VideoConfig:
    """Defaults for ``generate_video`` (provider-dependent)."""

    provider: Optional[str] = _opt(None, flat="video_provider")


# Maps the AixConfig attribute name -> (sub-config class, TOML section name).
# The attribute name doubles as the TOML section name.
_SECTIONS = {
    "chat": ChatConfig,
    "embeddings": EmbeddingConfig,
    "image": ImageConfig,
    "audio": AudioConfig,
    "video": VideoConfig,
}


@dataclass(frozen=True)
class AixConfig:
    """Top-level AIX configuration (single source of truth for defaults)."""

    chat: ChatConfig = field(default_factory=ChatConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    aliases: Mapping[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Loading (file + env layers)
# --------------------------------------------------------------------------- #


def config_file_path() -> str:
    """Path to the user config file (env ``AIX_CONFIG_FILE`` overrides default)."""
    explicit = os.environ.get("AIX_CONFIG_FILE")
    if explicit:
        return explicit
    # Imported lazily to avoid importing the data-dir machinery unless needed.
    from aix.util import djoin

    return djoin("config.toml")


def _read_toml(path: Optional[str]) -> dict:
    """Read a TOML config file, returning ``{}`` if missing or unreadable."""
    path = path or config_file_path()
    if not path or not os.path.isfile(path) or _tomllib is None:
        return {}
    with open(path, "rb") as f:
        return _tomllib.load(f)


def _build_subconfig(cls, section: dict, environ: Mapping[str, str]):
    """Build one sub-config: shipped defaults < TOML section < env vars."""
    kwargs = {}
    for f in fields(cls):
        toml_key = f.name
        if toml_key in section:
            kwargs[f.name] = section[toml_key]
        env = _env_name(f)
        if env in environ:
            kwargs[f.name] = f.metadata["cast"](environ[env])
    return cls(**kwargs)


def load_config(
    path: Optional[str] = None, *, environ: Optional[Mapping[str, str]] = None
) -> AixConfig:
    """Resolve an :class:`AixConfig` from shipped defaults, TOML file, and env.

    Precedence (low to high): shipped defaults < TOML file < environment variables.
    Runtime overrides (:func:`configure`/:func:`using`) and explicit call arguments
    sit above this and are applied elsewhere.

    Args:
        path: Optional explicit TOML path. Defaults to :func:`config_file_path`.
        environ: Optional environment mapping (defaults to ``os.environ``).

    Returns:
        A fully resolved :class:`AixConfig`.
    """
    environ = os.environ if environ is None else environ
    toml_data = _read_toml(path)

    sub = {
        attr: _build_subconfig(cls, toml_data.get(attr, {}) or {}, environ)
        for attr, cls in _SECTIONS.items()
    }
    aliases = dict(toml_data.get("aliases", {}) or {})
    return AixConfig(aliases=aliases, **sub)


# --------------------------------------------------------------------------- #
# Active config + runtime overrides
# --------------------------------------------------------------------------- #

_active_config: AixConfig = load_config()

# Flat override name -> (section attr, field name), built from field metadata.
_FLAT_INDEX = {
    f.metadata["flat"]: (attr, f.name)
    for attr, cls in _SECTIONS.items()
    for f in fields(cls)
}


def get_config() -> AixConfig:
    """Return the current active :class:`AixConfig`."""
    return _active_config


def set_config(config: AixConfig) -> AixConfig:
    """Replace the active config wholesale. Returns the new active config."""
    global _active_config
    if not isinstance(config, AixConfig):
        raise TypeError(f"Expected AixConfig, got {type(config)}")
    _active_config = config
    return _active_config


def _apply_overrides(base: AixConfig, overrides: Mapping[str, Any]) -> AixConfig:
    """Return a new AixConfig with flat ``overrides`` applied to ``base``.

    Keys are the flat names (e.g. ``chat_model``, ``tts_voice``) or top-level
    section names mapped to dicts (e.g. ``chat={'model': ...}``) or ``aliases``.
    """
    section_updates: dict[str, dict] = {}
    new_aliases = None
    for key, value in overrides.items():
        if key == "aliases":
            new_aliases = dict(value)
        elif key in _SECTIONS and isinstance(value, Mapping):
            section_updates.setdefault(key, {}).update(value)
        elif key in _FLAT_INDEX:
            attr, fname = _FLAT_INDEX[key]
            section_updates.setdefault(attr, {})[fname] = value
        else:
            raise TypeError(
                f"Unknown config override {key!r}. Valid flat keys: "
                f"{sorted(_FLAT_INDEX)}; or a section name in "
                f"{sorted(_SECTIONS)}; or 'aliases'."
            )

    changes: dict[str, Any] = {}
    for attr, updates in section_updates.items():
        changes[attr] = replace(getattr(base, attr), **updates)
    if new_aliases is not None:
        changes["aliases"] = new_aliases
    return replace(base, **changes)


def configure(**overrides: Any) -> AixConfig:
    """Apply runtime overrides to the active config and return it.

    Examples:
        >>> from aix import config
        >>> _ = config.configure(chat_model="openai/gpt-4o-mini")
        >>> config.get_config().chat.model
        'openai/gpt-4o-mini'
        >>> _ = config.configure(chat={"temperature": 0.2})
        >>> config.get_config().chat.temperature
        0.2
    """
    return set_config(_apply_overrides(_active_config, overrides))


@contextmanager
def using(**overrides: Any) -> Iterator[AixConfig]:
    """Context manager applying scoped overrides, restored on exit.

    Examples:
        >>> from aix import config
        >>> with config.using(chat_temperature=0.0) as c:
        ...     c.chat.temperature
        0.0
    """
    global _active_config
    previous = _active_config
    try:
        _active_config = _apply_overrides(previous, overrides)
        yield _active_config
    finally:
        _active_config = previous
