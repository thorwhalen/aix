# aix.structured

Structured output: a schema in, validated values out, enforced as far as the provider allows.

A *response schema* is a JSON Schema `dict` or a Pydantic model class. It is
turned into a [`ResponseSchema`](#aix.structured.ResponseSchema), which knows

* how to ask the provider for it – the strongest mode the model supports:
  - `"strict"`: `response_format={"type": "json_schema", ..., "strict": true}`,
    the provider constrains decoding to the schema. Used when LiteLLM reports
    `supports_response_schema(model)` *and* the schema is strict-compatible
    (every object closes `additionalProperties` and requires all its
    properties – Pydantic models are converted to that form automatically);
  - `"schema"`: the same `json_schema` format, non-strict, for a
    schema-capable model and a schema that is not strict-compatible;
  - `"json_mode"`: `{"type": "json_object"}` plus the schema in the prompt,
    for models without schema support (or whose support is unknown);
  - `"prompt"`: the schema in the prompt only, for models LiteLLM knows do
    not accept a `response_format` at all;
* and how to validate a reply client-side, which happens on every path (it is
  cheap, and “the provider enforced it” is a claim worth checking).

Providers only accept an *object* at the root of a structured reply (strict
`json_schema` and JSON mode alike), so a schema whose root is not an object
(an array, a union, a scalar) is sent wrapped as `{"value": <schema>}` and
unwrapped after validation – transparently to the caller.

```pycon
>>> spec = ResponseSchema.from_schema({"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]})
>>> spec.validate('{"x": 3}')
{'x': 3}
>>> spec.validate('{"x": "three"}')
Traceback (most recent call last):
...
aix.structured.SchemaValidationError: ...
```

### Classes

| [`ResponseSchema`](#aix.structured.ResponseSchema)(name, json_schema, ...[, ...])   | A normalized response schema: its JSON Schema, name, and validator.   |
|--------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|

### Exceptions

| [`SchemaValidationError`](#aix.structured.SchemaValidationError)(message, \*[, text, detail])   | A reply was not JSON, or did not match the response schema.   |
|-------------------------------------------------------------------------------------------------------|---------------------------------------------------------------|

### *class* aix.structured.ResponseSchema(name, json_schema, wire_schema, strict_compatible, source, wrapped=False, \_coerce_json=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A normalized response schema: its JSON Schema, name, and validator.

Build it with [`from_schema()`](#aix.structured.ResponseSchema.from_schema); [`aix.complete()`](aix.html.md#aix.complete) does that for you
from `response_schema=`.

#### name

The schema name sent to the provider.

#### json_schema

The caller’s JSON Schema dict (what a value must match).

#### wire_schema

What is sent: `json_schema` itself, or wrapped as
`{"value": json_schema}` when its root is not an object.

#### strict_compatible

Whether `wire_schema` satisfies strict-mode rules.

#### wrapped

Whether `wire_schema` is the `{"value": ...}` wrapper.

#### source

What the caller passed (a dict or a Pydantic model class).

#### *classmethod* from_schema(schema)

Normalize a JSON Schema dict or a Pydantic model class.

* **Raises:**
  * [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – For anything else.
  * **jsonschema.SchemaError** – For a malformed JSON Schema dict.
* **Return type:**
  [`ResponseSchema`](#aix.structured.ResponseSchema)

#### instruction()

The prompt text that asks for this schema (JSON / prompt modes, repairs).

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

#### mode_for(model)

The strongest enforcement mode available for `model` and this schema.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

#### response_format(mode)

The LiteLLM `response_format` for `mode` (`None` for `"prompt"`).

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

#### validate(text)

Parse and validate a reply; return the value (a model instance for Pydantic).

* **Raises:**
  [**SchemaValidationError**](#aix.structured.SchemaValidationError) – If `text` is not JSON or does not match.
* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

### *exception* aix.structured.SchemaValidationError(message, , text='', detail='')

Bases: [`ValueError`](https://docs.python.org/3/builtins/exceptions.html#ValueError)

A reply was not JSON, or did not match the response schema.

#### text

The reply that failed.

#### detail

The parse / validation message (what a repair prompt is told).
