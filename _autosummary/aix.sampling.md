# aix.sampling

Repeated sampling: `n` answers to one prompt, drawn as fast as the provider allows.

Asking the same question several times is the basic variance-reduction (and
stability-measurement) tool for LLM judgements. [`sample()`](#aix.sampling.sample) does it in one of
two ways and **records which**, because they are not the same statistical object:

* `"provider_n"` – one request with `n` choices (one prompt evaluation, `n`
  decodes). Cheapest, and used when LiteLLM reports the model supports `n`.
* `"independent"` – `n` separate requests, run concurrently. Works with every
  provider; each sample also gets its own `usage`.

If a provider silently returns fewer choices than asked, the shortfall is drawn
independently and the batch is marked `"mixed"`.

```pycon
>>> answers = sample("Is a tomato a fruit? Answer yes or no.", n=5)
>>> answers.texts
('Yes', 'Yes', 'Yes', 'yes', 'Yes')
>>> answers.strategy, answers.usage.total_tokens
('provider_n', 52)
```

### Functions

| [`sample`](#aix.sampling.sample)(prompt, \*, n[, temperature, seed, ...])   | Draw `n` answers to `prompt`, each with its finish reason and usage.      |
|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| [`concurrent_map`](#aix.sampling.concurrent_map)(func, items, \*, concurrency)      | `[func(x) for x in items]`, run on up to `concurrency` threads, in order. |
| [`check_positive_int`](#aix.sampling.check_positive_int)(name, value)                   | `int(value)` if it is a positive integer (not a bool), else `ValueError`. |

### Classes

| [`Sample`](#aix.sampling.Sample)(text, payload[, finish_reason, ...])   | One drawn answer.                                                                                                                                            |
|------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`Samples`](#aix.sampling.Samples)([samples, usage, n_requests, ...])    | The list of [`Sample`](#aix.sampling.Sample) returned by [`sample()`](#aix.sampling.sample), plus batch facts. |

### Exceptions

| [`SamplingError`](#aix.sampling.SamplingError)(message, \*[, samples])   | A request inside [`sample()`](#aix.sampling.sample) failed (with `on_error="raise"`).   |
|------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|

### *class* aix.sampling.Sample(text, payload, finish_reason=None, usage=None, strategy='independent', seed=None, error=None, system_fingerprint=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One drawn answer.

#### text

The raw text of this choice (`""` if its request failed).

#### payload

`parse(text)` when a `parse` function was given, else `text`;
`None` if the request or the parsing failed (see `error`).

#### finish_reason

Why this choice stopped. `"length"` = truncated,
`"content_filter"` = refused/filtered; check it before treating the
payload as a judgement.

#### usage

Tokens for the request that produced *only this* sample, or
`None` when it shared a provider-side `n` request (whose usage
cannot be split per choice; see [`Samples.usage`](#aix.sampling.Samples.usage)).

#### strategy

How this sample was drawn: `"provider_n"` or `"independent"`.

#### seed

The seed sent with its request, if any.

#### error

The exception this sample’s request raised (with
`on_error="record"`) or that `parse` raised on its text.

#### system_fingerprint

The serving backend’s fingerprint for its request,
when reported. Seeded samples are only comparable across equal
fingerprints (see [`aix.supports()`](aix.md#aix.supports) for whether `seed` is
honoured at all).

#### *property* ok *: [bool](https://docs.python.org/3/builtins/functions.html#bool)*

`True` when the request succeeded and the text parsed (if asked).

### *class* aix.sampling.Samples(samples=(), , strategy, usage=None, n_requests=0, n_usage_reports=0)

Bases: [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)

The list of [`Sample`](#aix.sampling.Sample) returned by [`sample()`](#aix.sampling.sample), plus batch facts.

It is a plain `list` otherwise: slicing or concatenating gives a `list`
without the batch attributes, which describe the whole draw only.

#### strategy

`"provider_n"`, `"independent"` or `"mixed"` (provider
`n` topped up with independent calls).

#### usage

Total tokens over every request that reported usage, or `None`
if none did. Compare `n_requests` with `n_usage_reports` to see
whether it is complete.

#### n_requests

How many requests were made (schema repair requests included).

#### n_usage_reports

How many of them reported usage (a completion’s repair
requests are counted with it when it reported usage).

#### *property* errors *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[Exception](https://docs.python.org/3/builtins/exceptions.html#Exception), ...]*

Every recorded error (request or parse), in sample order.

### *exception* aix.sampling.SamplingError(message, , samples=None)

Bases: [`Exception`](https://docs.python.org/3/builtins/exceptions.html#Exception)

A request inside [`sample()`](#aix.sampling.sample) failed (with `on_error="raise"`).

No request is launched after the first failure, and the ones already in
flight are awaited, so the work already paid for is not lost: `samples`
holds every request that was made, failed ones included (as samples whose
`error` is set) – possibly fewer than `n`. The first failure is
`__cause__`.

### aix.sampling.check_positive_int(name, value)

`int(value)` if it is a positive integer (not a bool), else `ValueError`.

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

```pycon
>>> check_positive_int("n", 3)
3
>>> check_positive_int("n", 0)
Traceback (most recent call last):
...
ValueError: n must be a positive integer, got 0
```

### aix.sampling.concurrent_map(func, items, , concurrency)

`[func(x) for x in items]`, run on up to `concurrency` threads, in order.

Fails fast like the serial loop would: once a call raises, calls not yet
started are cancelled (those already running finish), and the first
exception in item order propagates.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`R`)]

```pycon
>>> concurrent_map(lambda x: x * x, range(5), concurrency=3)
[0, 1, 4, 9, 16]
```

### aix.sampling.sample(prompt, , n, temperature=None, seed=None, concurrency=None, model=None, strategy='auto', parse=None, response_schema=None, on_error='raise', api_key=None, \*\*kwargs)

Draw `n` answers to `prompt`, each with its finish reason and usage.

* **Parameters:**
  * **prompt** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]) – A string or a list of message dicts, as for [`aix.chat()`](aix.md#aix.chat).
  * **n** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – How many samples to draw.
  * **temperature** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Sampling temperature (`None` = configured default). At
    `0` the samples mostly coincide, which is rarely what you want here.
  * **seed** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Base seed. A provider-side `n` request sends it as is; each
    independent request sends a seed derived from `(seed, index)` so
    independent samples are not forced identical and runs with different
    base seeds do not overlap. Support is provider-dependent.
  * **concurrency** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Max simultaneous independent requests (`None` = config
    `chat.sample_concurrency`).
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Model id or alias (`None` = configured default).
  * **strategy** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'auto'`, `'provider'`, `'independent'`]) – `"auto"` (provider `n` when supported, else independent;
    also falls back to independent if the provider refuses `n`),
    `"provider"` (force one `n` request) or `"independent"`.
    `n=1` is always a single plain request, recorded `"independent"`.
  * **response_schema** (`Union`[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict), [`type`](https://docs.python.org/3/builtins/functions.html#type)]) – A JSON Schema dict or Pydantic model class, enforced and
    validated per request as in [`aix.complete()`](aix.md#aix.complete); each payload is the
    validated value, and a choice still invalid after repairs records its
    `SchemaValidationError` as `error`. Exclusive with `parse`.
  * **parse** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – Optional `text -> payload` (e.g. `json.loads`). A failure is
    recorded on that sample (`payload=None`, `error=...`), not raised.
  * **on_error** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'raise'`, `'record'`]) – When a request fails: `"raise"` (default) raises
    [`SamplingError`](#aix.sampling.SamplingError) once the launched requests finish, carrying
    the partial batch; `"record"` returns the batch with the failure on
    that sample’s `error`.
  * **api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Explicit API key, as for [`aix.chat()`](aix.md#aix.chat).
  * **\*\*kwargs** – Passed to every request (e.g. `max_tokens`, `response_format`).
* **Return type:**
  [`Samples`](#aix.sampling.Samples)
* **Returns:**
  [`Samples`](#aix.sampling.Samples) – a list of [`Sample`](#aix.sampling.Sample) with batch `.strategy`,
  `.usage`, `.n_requests`.
* **Raises:**
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – For a non-positive `n` / `concurrency`, an unknown
        `strategy` / `on_error`, or a reserved kwarg (`messages`, `stream`).
  * [**SamplingError**](#aix.sampling.SamplingError) – A request failed and `on_error="raise"`.
