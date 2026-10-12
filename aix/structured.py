"""Structured output: a schema in, validated values out, enforced as far as the provider allows.

A *response schema* is a JSON Schema ``dict`` or a Pydantic model class. It is
turned into a :class:`ResponseSchema`, which knows

* how to ask the provider for it -- the strongest mode the model supports:

  - ``"strict"``: ``response_format={"type": "json_schema", ..., "strict": true}``,
    the provider constrains decoding to the schema. Used when LiteLLM reports
    ``supports_response_schema(model)`` *and* the schema is strict-compatible
    (every object closes ``additionalProperties`` and requires all its
    properties -- Pydantic models are converted to that form automatically);
  - ``"schema"``: the same ``json_schema`` format, non-strict, for a
    schema-capable model and a schema that is not strict-compatible;
  - ``"json_mode"``: ``{"type": "json_object"}`` plus the schema in the prompt,
    for models without schema support;

* and how to validate a reply client-side, which happens on every path (it is
  cheap, and "the provider enforced it" is a claim worth checking).

>>> spec = ResponseSchema.from_schema({"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]})
>>> spec.validate('{"x": 3}')
{'x': 3}
>>> spec.validate('{"x": "three"}')
Traceback (most recent call last):
...
aix.structured.SchemaValidationError: ...
"""

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from aix._litellm import UNRESOLVED as _UNRESOLVED, litellm_attr as _litellm_attr

__all__ = [
    "ResponseSchema",
    "SchemaValidationError",
    "STRICT",
    "SCHEMA",
    "JSON_MODE",
]

STRICT = "strict"
SCHEMA = "schema"
JSON_MODE = "json_mode"

_DFLT_SCHEMA_NAME = "response"
# OpenAI-compatible APIs restrict json_schema names to this alphabet / length.
_SCHEMA_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9_-]")
_SCHEMA_NAME_MAX_LEN = 64
_FENCE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)

_litellm_supports_response_schema = _UNRESOLVED


class SchemaValidationError(ValueError):
    """A reply was not JSON, or did not match the response schema.

    Attributes:
        text: The reply that failed.
        detail: The parse / validation message (what a repair prompt is told).
    """

    def __init__(self, message: str, *, text: str = "", detail: str = ""):
        super().__init__(message)
        self.text = text
        self.detail = detail or message


def model_supports_response_schema(model: str) -> bool:
    """Whether LiteLLM says ``model`` accepts a ``json_schema`` response format.

    ``False`` when unknown (or litellm is missing), so callers degrade to JSON mode.
    """
    global _litellm_supports_response_schema
    if _litellm_supports_response_schema is _UNRESOLVED:
        _litellm_supports_response_schema = _litellm_attr("supports_response_schema")
    if _litellm_supports_response_schema is None:
        return False
    try:
        return bool(_litellm_supports_response_schema(model=model))
    except Exception:
        return False


def _is_pydantic_model(obj: Any) -> bool:
    return isinstance(obj, type) and hasattr(obj, "model_json_schema")


def _is_strict_compatible(schema: Any) -> bool:
    """Whether every object in ``schema`` is closed and requires all its properties.

    That is what OpenAI-style ``strict: true`` demands; sending a schema that
    is not, makes the provider reject the request rather than the reply.

    >>> _is_strict_compatible({"type": "object", "properties": {"a": {"type": "string"}},
    ...                        "required": ["a"], "additionalProperties": False})
    True
    >>> _is_strict_compatible({"type": "object", "properties": {"a": {"type": "string"}}})
    False
    """
    if isinstance(schema, Mapping):
        if schema.get("type") == "object" or "properties" in schema:
            props = schema.get("properties", {})
            if schema.get("additionalProperties", True) is not False:
                return False
            if set(schema.get("required", ())) != set(props):
                return False
        return all(_is_strict_compatible(v) for v in schema.values())
    if isinstance(schema, list):
        return all(_is_strict_compatible(v) for v in schema)
    return True


def _schema_name(raw: str) -> str:
    name = _SCHEMA_NAME_PATTERN.sub("_", raw)[:_SCHEMA_NAME_MAX_LEN]
    return name or _DFLT_SCHEMA_NAME


def _loads_json(text: str) -> Any:
    """Parse a JSON reply, tolerating a surrounding markdown code fence."""
    match = _FENCE.match(text)
    return json.loads(match.group(1) if match else text)


def _jsonschema_validator(schema: Mapping) -> Callable[[Any], Any]:
    """A ``value -> value`` validator for a JSON Schema dict (raises ValueError)."""
    try:
        import jsonschema
    except ImportError as e:  # pragma: no cover - jsonschema ships with litellm
        raise ImportError(
            "Validating a JSON Schema response_schema needs `jsonschema` "
            "(installed with litellm): pip install jsonschema"
        ) from e
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)  # a malformed schema fails at definition
    validator = validator_cls(schema)

    def validate(value):
        errors = sorted(validator.iter_errors(value), key=lambda e: list(e.path))
        if errors:
            path = "/".join(map(str, errors[0].absolute_path)) or "<root>"
            raise ValueError(f"at {path}: {errors[0].message}")
        return value

    return validate


@dataclass(frozen=True)
class ResponseSchema:
    """A normalized response schema: its JSON Schema, name, and validator.

    Build it with :meth:`from_schema`; :func:`aix.complete` does that for you
    from ``response_schema=``.

    Attributes:
        name: The schema name sent to the provider.
        json_schema: The JSON Schema dict.
        strict_compatible: Whether ``json_schema`` satisfies strict-mode rules.
        source: What the caller passed (a dict or a Pydantic model class).
    """

    name: str
    json_schema: dict
    strict_compatible: bool
    source: Any = field(repr=False, compare=False)
    _coerce: Callable[[Any], Any] = field(repr=False, compare=False, default=None)

    @classmethod
    def from_schema(
        cls, schema: Union[Mapping, type, "ResponseSchema"]
    ) -> "ResponseSchema":
        """Normalize a JSON Schema dict or a Pydantic model class.

        Raises:
            TypeError: For anything else.
            jsonschema.SchemaError: For a malformed JSON Schema dict.
        """
        if isinstance(schema, ResponseSchema):
            return schema
        if _is_pydantic_model(schema):
            return cls._from_pydantic(schema)
        if isinstance(schema, Mapping):
            schema = dict(schema)
            return cls(
                name=_schema_name(str(schema.get("title", _DFLT_SCHEMA_NAME))),
                json_schema=schema,
                strict_compatible=_is_strict_compatible(schema),
                source=schema,
                _coerce=_jsonschema_validator(schema),
            )
        raise TypeError(
            "response_schema must be a JSON Schema dict or a Pydantic model class, "
            f"got {type(schema).__name__}"
        )

    @classmethod
    def _from_pydantic(cls, model: type) -> "ResponseSchema":
        to_format = _litellm_attr("utils")
        to_format = getattr(to_format, "type_to_response_format_param", None)
        if to_format is not None:
            # litellm closes objects / requires all fields, i.e. strict form.
            spec = to_format(model)["json_schema"]
            json_schema, name = spec["schema"], spec.get("name", model.__name__)
        else:  # pragma: no cover - older litellm
            json_schema, name = model.model_json_schema(), model.__name__
        return cls(
            name=_schema_name(name),
            json_schema=json_schema,
            strict_compatible=_is_strict_compatible(json_schema),
            source=model,
            _coerce=model.model_validate,
        )

    def response_format(self, mode: str) -> dict:
        """The LiteLLM ``response_format`` for ``mode`` (see the module docs)."""
        if mode == JSON_MODE:
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self.name,
                "schema": self.json_schema,
                "strict": mode == STRICT,
            },
        }

    def mode_for(self, model: str) -> str:
        """The strongest enforcement mode available for ``model`` and this schema."""
        if not model_supports_response_schema(model):
            return JSON_MODE
        return STRICT if self.strict_compatible else SCHEMA

    def instruction(self) -> str:
        """The prompt text that asks for this schema (used in JSON mode, and repairs)."""
        return (
            "Respond with a single JSON value that validates against this JSON "
            f"Schema, and nothing else:\n{json.dumps(self.json_schema)}"
        )

    def validate(self, text: str) -> Any:
        """Parse and validate a reply; return the value (a model instance for Pydantic).

        Raises:
            SchemaValidationError: If ``text`` is not JSON or does not match.
        """
        try:
            value = _loads_json(text)
        except (json.JSONDecodeError, TypeError) as e:
            raise SchemaValidationError(
                f"Reply is not valid JSON: {e}", text=text, detail=f"not JSON ({e})"
            ) from e
        try:
            return self._coerce(value)
        except Exception as e:  # pydantic.ValidationError / ValueError
            raise SchemaValidationError(
                f"Reply does not match the response schema: {e}",
                text=text,
                detail=str(e),
            ) from e
