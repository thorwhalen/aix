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
    for models without schema support (or whose support is unknown);
  - ``"prompt"``: the schema in the prompt only, for models LiteLLM knows do
    not accept a ``response_format`` at all;

* and how to validate a reply client-side, which happens on every path (it is
  cheap, and "the provider enforced it" is a claim worth checking).

Providers only accept an *object* at the root of a structured reply (strict
``json_schema`` and JSON mode alike), so a schema whose root is not an object
(an array, a union, a scalar) is sent wrapped as ``{"value": <schema>}`` and
unwrapped after validation -- transparently to the caller.

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
    "PROMPT",
]

STRICT = "strict"
SCHEMA = "schema"
JSON_MODE = "json_mode"
PROMPT = "prompt"

# Root key used to wrap a non-object schema (see the module docs).
_WRAP_KEY = "value"
# Keywords OpenAI-style strict mode has rejected; a schema using them is sent
# non-strict (still schema-guided, still validated client-side). Conservative:
# a false "not strict" costs little, a false "strict" fails the request.
_STRICT_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
        "patternProperties",
        "prefixItems",
        "if",
        "then",
        "else",
        "not",
        "dependentRequired",
        "dependentSchemas",
    }
)
# Sub-schema positions, walked structurally (never into enum/const/default data).
_SUBSCHEMA_MAPS = ("properties", "$defs", "definitions")
_SUBSCHEMA_LISTS = ("anyOf", "oneOf", "allOf", "prefixItems")
_SUBSCHEMA_SINGLE = ("items", "additionalProperties", "not", "if", "then", "else")

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


def _subschemas(node: Mapping):
    """The direct sub-schemas of a schema node (structural positions only)."""
    for key in _SUBSCHEMA_MAPS:
        if isinstance(node.get(key), Mapping):
            yield from node[key].values()
    for key in _SUBSCHEMA_LISTS:
        if isinstance(node.get(key), list):
            yield from node[key]
    for key in _SUBSCHEMA_SINGLE:
        value = node.get(key)
        if isinstance(value, Mapping):
            yield value
        elif key == "items" and isinstance(value, list):
            yield from value


def _is_object_node(node: Mapping) -> bool:
    kind = node.get("type")
    is_object_type = kind == "object" or (isinstance(kind, list) and "object" in kind)
    return is_object_type or "properties" in node


def _is_object_root(schema: Mapping) -> bool:
    """Whether ``schema`` can only be an object (what providers accept at the root)."""
    kind = schema.get("type")
    return kind == "object" or (kind is None and "properties" in schema)


def _is_strict_compatible(schema: Any) -> bool:
    """Whether ``schema`` meets OpenAI-style ``strict: true`` rules.

    Every object node closes ``additionalProperties`` and requires all of its
    properties, and no node uses a keyword strict mode has rejected. Sending a
    schema that is not, makes the provider reject the request (not the reply).
    Walks sub-schema positions only, so data such as ``enum``/``default``
    values -- or a property that happens to be named ``properties`` -- is never
    mistaken for a schema.

    >>> _is_strict_compatible({"type": "object", "properties": {"a": {"type": "string"}},
    ...                        "required": ["a"], "additionalProperties": False})
    True
    >>> _is_strict_compatible({"type": "object", "properties": {"a": {"type": "string"}}})
    False
    """
    if not isinstance(schema, Mapping):
        return not isinstance(schema, bool)  # boolean schemas are not strict
    if _STRICT_UNSUPPORTED_KEYWORDS.intersection(schema):
        return False
    if _is_object_node(schema):
        props = schema.get("properties", {})
        if schema.get("additionalProperties", True) is not False:
            return False
        if set(schema.get("required", ())) != set(props):
            return False
    return all(_is_strict_compatible(sub) for sub in _subschemas(schema))


def _wrapped(schema: dict) -> dict:
    """``{"value": schema}`` as a closed object, hoisting ``$defs`` so refs resolve."""
    inner = dict(schema)
    defs = {k: inner.pop(k) for k in ("$defs", "definitions") if k in inner}
    return {
        "type": "object",
        "properties": {_WRAP_KEY: inner},
        "required": [_WRAP_KEY],
        "additionalProperties": False,
        **defs,
    }


def _schema_name(raw: str) -> str:
    name = _SCHEMA_NAME_PATTERN.sub("_", raw)[:_SCHEMA_NAME_MAX_LEN]
    return name or _DFLT_SCHEMA_NAME


def _unfenced(text: str) -> str:
    """``text`` without a surrounding markdown code fence."""
    match = _FENCE.match(text)
    return match.group(1) if match else text


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
        json_schema: The caller's JSON Schema dict (what a value must match).
        wire_schema: What is sent: ``json_schema`` itself, or wrapped as
            ``{"value": json_schema}`` when its root is not an object.
        strict_compatible: Whether ``wire_schema`` satisfies strict-mode rules.
        wrapped: Whether ``wire_schema`` is the ``{"value": ...}`` wrapper.
        source: What the caller passed (a dict or a Pydantic model class).
    """

    name: str
    json_schema: dict
    wire_schema: dict
    strict_compatible: bool
    source: Any = field(repr=False, compare=False)
    wrapped: bool = False
    # ``json text -> value`` (raises on invalid); see _from_dict / _from_pydantic.
    _coerce_json: Callable[[str], Any] = field(repr=False, compare=False, default=None)

    @classmethod
    def _build(cls, name: str, json_schema: dict, source, coerce_json):
        wrapped = not _is_object_root(json_schema)
        wire = _wrapped(json_schema) if wrapped else json_schema
        return cls(
            name=_schema_name(name),
            json_schema=json_schema,
            wire_schema=wire,
            strict_compatible=_is_strict_compatible(wire),
            wrapped=wrapped,
            source=source,
            _coerce_json=coerce_json,
        )

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
            validate = _jsonschema_validator(schema)
            return cls._build(
                str(schema.get("title", _DFLT_SCHEMA_NAME)),
                schema,
                schema,
                lambda text: validate(json.loads(text)),
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
        # JSON-mode validation, so enums / dates / strict-config models accept
        # the strings a JSON reply necessarily carries.
        return cls._build(name, json_schema, model, model.model_validate_json)

    def response_format(self, mode: str) -> Optional[dict]:
        """The LiteLLM ``response_format`` for ``mode`` (``None`` for ``"prompt"``)."""
        if mode == PROMPT:
            return None
        if mode == JSON_MODE:
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self.name,
                "schema": self.wire_schema,
                "strict": mode == STRICT,
            },
        }

    def mode_for(self, model: str) -> str:
        """The strongest enforcement mode available for ``model`` and this schema."""
        if model_supports_response_schema(model):
            return STRICT if self.strict_compatible else SCHEMA
        from aix.capabilities import supported_params

        params = supported_params(model)
        if params is not None and "response_format" not in params:
            return PROMPT  # known not to take any response_format
        return JSON_MODE

    def instruction(self) -> str:
        """The prompt text that asks for this schema (JSON / prompt modes, repairs)."""
        return (
            "Respond with a single JSON object that validates against this JSON "
            f"Schema, and nothing else:\n{json.dumps(self.wire_schema)}"
        )

    def validate(self, text: str) -> Any:
        """Parse and validate a reply; return the value (a model instance for Pydantic).

        Raises:
            SchemaValidationError: If ``text`` is not JSON or does not match.
        """
        payload = _unfenced(text or "")
        try:
            parsed = json.loads(payload)
        except (json.JSONDecodeError, TypeError) as e:
            raise SchemaValidationError(
                f"Reply is not valid JSON: {e}", text=text, detail=f"not JSON ({e})"
            ) from e
        if self.wrapped:
            if not (isinstance(parsed, dict) and set(parsed) == {_WRAP_KEY}):
                raise SchemaValidationError(
                    f"Reply is not a {{{_WRAP_KEY!r}: ...}} object",
                    text=text,
                    detail=f'expected an object with the single key "{_WRAP_KEY}"',
                )
            payload = json.dumps(parsed[_WRAP_KEY])
        try:
            return self._coerce_json(payload)
        except Exception as e:  # pydantic.ValidationError / ValueError
            raise SchemaValidationError(
                f"Reply does not match the response schema: {e}",
                text=text,
                detail=str(e),
            ) from e
