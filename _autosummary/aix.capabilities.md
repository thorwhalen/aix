# aix.capabilities

What a model supports, asked before calling it.

Providers differ in which OpenAI-style parameters they honour (`n`, `seed`,
`response_format`…). LiteLLM knows, per model, through
`get_supported_openai_params`; this module puts a small, total (never
raising) question in front of it, so callers can *choose* a degraded path
rather than discover one silently.

Besides request parameter names, [`supports()`](#aix.capabilities.supports) answers a few *capabilities*
that are not a single parameter (see [`CAPABILITIES`](#aix.capabilities.CAPABILITIES)), e.g.
`"response_schema"`: whether the model takes a JSON-Schema-constrained
`response_format` (what `aix.complete(..., response_schema=...)` uses for
its `"strict"` / `"schema"` modes).

Why ask about `seed` in particular: a run-to-run difference with a seed the
provider *ignored* means nothing about model stability. `supports(model,
"seed") is False` rules a seed out; `True` only says LiteLLM forwards it –
seeded sampling is best-effort on every provider, and a router may drop it
upstream (compare
[`aix.Completion.system_fingerprint`](aix.md#aix.Completion.system_fingerprint) across runs: a change means the
serving backend changed, which voids the comparison).

```pycon
>>> supports("gpt-4o-mini", "n")
True
>>> supports("anthropic/claude-sonnet-4-20250514", "n")
False
```

### Module Attributes

| [`CAPABILITIES`](#aix.capabilities.CAPABILITIES)   | Capabilities [`supports()`](#aix.capabilities.supports) answers that are not one request parameter, mapped to the `model -> bool` check behind each.   |
|-----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

### Functions

| [`supports`](#aix.capabilities.supports)(model, param)    | Whether `model` is known to accept `param` -- i.e. whether LiteLLM *forwards* it to the provider for this model.   |
|----------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| [`supported_params`](#aix.capabilities.supported_params)([model]) | The OpenAI-style parameters LiteLLM says `model` accepts.                                                          |

### aix.capabilities.CAPABILITIES *= {'response_schema': <function \_supports_response_schema>}*

Capabilities [`supports()`](#aix.capabilities.supports) answers that are not one request parameter,
mapped to the `model -> bool` check behind each.

### aix.capabilities.supported_params(model=None)

The OpenAI-style parameters LiteLLM says `model` accepts.

`None` when that is unknown (LiteLLM cannot place the model, or is not
installed) – which is not the same as “supports nothing”.

* **Parameters:**
  **model** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Model id or alias; `None` means the configured chat default.
* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`frozenset`](https://docs.python.org/3/builtins/stdtypes.html#frozenset)]

### aix.capabilities.supports(model, param)

Whether `model` is known to accept `param` – i.e. whether LiteLLM
*forwards* it to the provider for this model.

Conservative: `False` when support is unknown (or for a non-string
`param`), so a caller that branches on it takes the path that works
everywhere. `True` is not a guarantee that the parameter has an effect:
a router (OpenRouter…) can forward `seed` / `n` to an upstream model
that drops them, and a provider can deprecate or ignore a parameter
(seeding is best-effort everywhere; reasoning models may ignore it).

* **Parameters:**
  * **model** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Model id or alias; `None` means the configured chat default.
  * **param** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – An OpenAI-style request parameter name (`"n"`, `"seed"`,
    `"response_format"`…) or a capability in [`CAPABILITIES`](#aix.capabilities.CAPABILITIES)
    (`"response_schema"`).
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

```pycon
>>> supports("gpt-4o-mini", "seed")
True
>>> supports("gpt-4o-mini", "response_schema")
True
```
