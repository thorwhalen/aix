# aix.capabilities

What a model supports, asked before calling it.

Providers differ in which OpenAI-style parameters they honour (`n`, `seed`,
`response_format`…). LiteLLM knows, per model, through
`get_supported_openai_params`; this module puts a small, total (never
raising) question in front of it, so callers can *choose* a degraded path
rather than discover one silently.

```pycon
>>> supports("gpt-4o-mini", "n")
True
>>> supports("anthropic/claude-sonnet-4-20250514", "n")
False
```

### Functions

| [`supports`](#aix.capabilities.supports)(model, param)    | Whether `model` is known to accept the request parameter `param`.   |
|----------------------------------------------------------------------------|---------------------------------------------------------------------|
| [`supported_params`](#aix.capabilities.supported_params)([model]) | The OpenAI-style parameters LiteLLM says `model` accepts.           |

### aix.capabilities.supported_params(model=None)

The OpenAI-style parameters LiteLLM says `model` accepts.

`None` when that is unknown (LiteLLM cannot place the model, or is not
installed) – which is not the same as “supports nothing”.

* **Parameters:**
  **model** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Model id or alias; `None` means the configured chat default.
* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`frozenset`](https://docs.python.org/3/builtins/stdtypes.html#frozenset)]

### aix.capabilities.supports(model, param)

Whether `model` is known to accept the request parameter `param`.

Conservative: `False` when support is unknown, so a caller that branches
on it takes the path that works everywhere.

* **Parameters:**
  * **model** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Model id or alias; `None` means the configured chat default.
  * **param** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – An OpenAI-style request parameter name (`"n"`, `"seed"`…).
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
