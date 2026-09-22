# aix.vision

Image-to-text (vision) interface for AIX.

The cross-modal counterpart to [`aix.image`](aix.image.md#module-aix.image) (which is text→image): given an
image, produce text — a caption, an answer to a question about it, or a
structured judgement. Like the rest of AIX it is a thin, provider-neutral facade
over LiteLLM’s multimodal `completion` (the same backend `chat` uses), so the
same call routes to any vision-capable provider (OpenAI, Anthropic, Gemini,
OpenRouter, …) by model id alone.

[`describe_image()`](#aix.vision.describe_image) is the primitive: image (URL / local path / bytes / PIL
image / `data:` URI) + a prompt → text. [`to_image_content()`](#aix.vision.to_image_content) exposes the
multimodal content block builder so callers can assemble richer messages (e.g.
multiple images in one turn) and pass them straight to [`aix.chat()`](aix.md#aix.chat).

### Examples

Caption an image (any vision-capable model):

```pycon
>>> from aix.vision import describe_image
>>> describe_image("https://example.com/cat.jpg")
'A grey tabby cat sitting on a windowsill in afternoon light.'
```

Ask a specific question:

```pycon
>>> describe_image("photo.png", prompt="What colour is the car?")
'The car is red.'
```

Build a multi-image message yourself and hand it to `chat`:

```pycon
>>> from aix import chat
>>> from aix.vision import to_image_content
>>> msg = [{"role": "user", "content": [
...     {"type": "text", "text": "Which is brighter?"},
...     to_image_content("a.jpg"), to_image_content("b.jpg"),
... ]}]
>>> chat(msg, model="gpt-4o")
'The second image is brighter.'
```

[`compare_images()`](#aix.vision.compare_images) builds on these primitives: it puts a *candidate* image
beside one or more *reference* images and asks a vision model, via a structured
JSON path, for an explainable per-aspect likeness verdict — the “explain the
drift” half of a reference supervisor (face / costume / setting / lighting /
props matched?). The numeric identity-cosine gate lives elsewhere (lookbook);
this layer is the explainable checklist on top of it.

```pycon
>>> from aix.vision import compare_images
>>> verdict = compare_images("gen.png", "locked_ref.png")
>>> verdict.match, verdict["identity"].match
(True, True)
```

### Module Attributes

| [`DFLT_VISION_MODEL`](#aix.vision.DFLT_VISION_MODEL)   | Shipped-default vision model, kept for reference.                                                                    |
|----------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| [`DFLT_VISION_PROMPT`](#aix.vision.DFLT_VISION_PROMPT)  | Default instruction when the caller doesn't supply one.                                                              |
| [`DFLT_COMPARE_RUBRIC`](#aix.vision.DFLT_COMPARE_RUBRIC) | Default rubric for [`compare_images()`](#aix.vision.compare_images) — a generic likeness checklist. |
| [`DFLT_FILM_RUBRIC`](#aix.vision.DFLT_FILM_RUBRIC)    | the locked face, the architecture/set, the lighting, and the props.                                                  |

### Functions

| [`describe_image`](#aix.vision.describe_image)(image, \*[, prompt, model, ...])   | Describe (or answer a question about) `image` and return the text.   |
|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| [`to_image_content`](#aix.vision.to_image_content)(image, \*[, detail])             | Build a multimodal `image_url` content block for `image`.            |
| [`compare_images`](#aix.vision.compare_images)(candidate, reference, \*[, ...])   | Compare a `candidate` image to `reference` image(s) on a rubric.     |

### Classes

| [`RubricVerdict`](#aix.vision.RubricVerdict)(aspect, match, confidence[, note])   | A vision model's verdict on a single rubric aspect.               |
|-----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| [`ImageComparison`](#aix.vision.ImageComparison)(match, confidence, explanation)    | Structured result of comparing a candidate image to reference(s). |

### aix.vision.DFLT_COMPARE_RUBRIC *= ('identity', 'costume', 'setting', 'lighting', 'props')*

Default rubric for [`compare_images()`](#aix.vision.compare_images) — a generic likeness checklist.
Each entry is one aspect the vision model judges independently.

### aix.vision.DFLT_FILM_RUBRIC *= ('face_identity', 'costume', 'setting_architecture', 'lighting', 'props', 'skin_realism')*

the
locked face, the architecture/set, the lighting, and the props. Pass it as
`rubric=` when comparing a generated frame against a locked reference.

* **Type:**
  Filmmaking-oriented rubric (Noel’s reference-supervisor checklist)

### aix.vision.DFLT_VISION_MODEL *= 'gpt-4o-mini'*

Shipped-default vision model, kept for reference. The *active* default is
resolved from `aix.config` at call time (see [`aix.config.VisionConfig`](aix.config.md#aix.config.VisionConfig)).

### aix.vision.DFLT_VISION_PROMPT *= 'Describe this image in detail.'*

Default instruction when the caller doesn’t supply one.

### *class* aix.vision.ImageComparison(match, confidence, explanation, aspects=<factory>, model=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Structured result of comparing a candidate image to reference(s).

Behaves like an ordered, read-only mapping of `aspect -> RubricVerdict`
(`comparison["identity"]`, `in`, iteration, `len`) so per-aspect
lookups read naturally, while also carrying the overall verdict.

#### match

Overall pass/fail across the whole rubric.

#### confidence

Overall confidence in `[0.0, 1.0]`.

#### explanation

A short overall summary of the comparison.

#### aspects

The per-aspect verdicts, one [`RubricVerdict`](#aix.vision.RubricVerdict) per
rubric item, in rubric order.

#### model

The vision model id that produced the verdict.

#### get(aspect, default=None)

Return the verdict for `aspect`, or `default` if absent.

* **Return type:**
  [`RubricVerdict`](#aix.vision.RubricVerdict) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

### *class* aix.vision.RubricVerdict(aspect, match, confidence, note='')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A vision model’s verdict on a single rubric aspect.

#### aspect

The rubric item this verdict is about (e.g. `"identity"`).

#### match

Whether the candidate matches the reference for this aspect.

#### confidence

The model’s self-reported confidence in `[0.0, 1.0]`.

#### note

A short free-text explanation of the verdict.

### aix.vision.compare_images(candidate, reference, , rubric=('identity', 'costume', 'setting', 'lighting', 'props'), model=None, api_key=None, max_tokens=None, temperature=0.0, detail=None, instruction='You are a strict visual continuity supervisor. The FIRST image is the CANDIDATE; the remaining image(s) are the locked REFERENCE the candidate must match. Judge each rubric aspect INDEPENDENTLY (do not let one aspect color another). For each aspect decide whether the candidate matches the reference, give a confidence in [0.0, 1.0], and a short note explaining what matches or drifts. Then give an overall match (true only if every important aspect matches), an overall confidence, and a one-sentence explanation.', \*\*kwargs)

Compare a `candidate` image to `reference` image(s) on a rubric.

The explainable half of a reference supervisor: a vision model returns a
per-aspect pass/fail checklist (does the face match? the costume? the set?
the lighting? the props?) plus an overall verdict — the explainable layer
over a cheap numeric identity-cosine gate (which lives elsewhere, e.g.
lookbook). Built on [`to_image_content()`](#aix.vision.to_image_content) (multi-image content block)
and the same multimodal `completion` path as [`describe_image()`](#aix.vision.describe_image),
asked via a JSON contract for a structured answer.

The comparison is *pointwise* (each aspect judged on its own), not a ranked
pairwise comparison, to avoid position bias.

* **Parameters:**
  * **candidate** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – The image under review (URL / path / bytes / PIL image /
    `data:` URI — anything [`to_image_content()`](#aix.vision.to_image_content) accepts).
  * **reference** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any), [`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – The locked reference — a single image *or* a sequence of
    images (a locked set) in the same flexible formats. An empty
    sequence is an error.
  * **rubric** ([`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – The aspects to evaluate, one verdict per item. Defaults to
    [`DFLT_COMPARE_RUBRIC`](#aix.vision.DFLT_COMPARE_RUBRIC); pass [`DFLT_FILM_RUBRIC`](#aix.vision.DFLT_FILM_RUBRIC) (or any
    custom sequence) to override — e.g. `("face", "architecture",
    "props", "lighting")`. Must be non-empty.
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Vision-capable model id or alias. `None` → the configured
    default ([`aix.config.VisionConfig`](aix.config.md#aix.config.VisionConfig)).
  * **api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Explicit API key; `None` resolves it for the model’s
    provider from the environment / `.env` / config store.
  * **max_tokens** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Cap on generated tokens (`None` → provider default).
  * **temperature** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Sampling temperature (default `0.0` for a stable,
    reproducible verdict; `None` → provider default).
  * **detail** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Image detail hint (`"low"` | `"high"` | `"auto"`),
    applied to every image block.
  * **instruction** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The system-style framing prepended to the rubric. Has a
    sensible default; override to retune the supervisor’s strictness.
  * **\*\*kwargs** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Extra provider-specific params forwarded to LiteLLM.
* **Return type:**
  [`ImageComparison`](#aix.vision.ImageComparison)
* **Returns:**
  An [`ImageComparison`](#aix.vision.ImageComparison) — overall `match` / `confidence` /
  `explanation` plus an ordered, mapping-like collection of
  [`RubricVerdict`](#aix.vision.RubricVerdict) (one per rubric aspect, keyed by aspect).
* **Raises:**
  * [**ImportError**](https://docs.python.org/3/builtins/exceptions.html#ImportError) – If LiteLLM is not installed.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If `rubric` is empty or `reference` is an empty
        sequence, or if the model’s reply can’t be parsed as the expected
        JSON verdict.

### Examples

Default rubric, single reference:

```pycon
>>> compare_images("gen.png", "ref.png")
ImageComparison(match=True, confidence=0.9, ...)
```

Filmmaking rubric, a locked reference *set*:

```pycon
>>> from aix.vision import DFLT_FILM_RUBRIC
>>> compare_images(
...     candidate="frame_042.png",
...     reference=["ref_front.png", "ref_side.png"],
...     rubric=DFLT_FILM_RUBRIC,
... )["face_identity"].match
True
```

### aix.vision.describe_image(image, , prompt='Describe this image in detail.', model=None, api_key=None, max_tokens=None, temperature=None, detail=None, \*\*kwargs)

Describe (or answer a question about) `image` and return the text.

The image→text primitive. `image` accepts a URL, a local path, raw bytes,
a PIL image, or a `data:` URI (see [`to_image_content()`](#aix.vision.to_image_content)). `prompt` is
the instruction (default: a generic “describe this image”); pass a question
for VQA or a rubric for a judgement. Everything past `image` is keyword.

* **Parameters:**
  * **image** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – The image (URL / path / bytes / PIL image / `data:` URI).
  * **prompt** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The text instruction accompanying the image.
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Vision-capable model id (e.g. `gpt-4o`, `claude-sonnet-4-6`,
    `gemini/gemini-1.5-pro`) or an alias. `None` → the configured
    default ([`aix.config.VisionConfig`](aix.config.md#aix.config.VisionConfig)).
  * **api_key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Explicit API key; `None` resolves it from the environment /
    `.env` / config store for the model’s provider.
  * **max_tokens** ([`int`](https://docs.python.org/3/builtins/functions.html#int) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Cap on generated tokens (`None` → provider default).
  * **temperature** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Sampling temperature (`None` → provider default).
  * **detail** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Image detail hint (`"low"` | `"high"` | `"auto"`).
  * **\*\*kwargs** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Extra provider-specific params forwarded to LiteLLM.
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  The model’s text response.
* **Raises:**
  [**ImportError**](https://docs.python.org/3/builtins/exceptions.html#ImportError) – If LiteLLM is not installed.

```pycon
>>> describe_image("cat.jpg", prompt="Caption it.")
'A cat on a sofa.'
```

### aix.vision.to_image_content(image, , detail=None)

Build a multimodal `image_url` content block for `image`.

`image` may be:

- an `http(s)://` URL or a `data:` URI — passed through verbatim;
- a local file path (`str` or [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)) — read and inlined
  as a base64 `data:` URI with a guessed MIME type;
- raw `bytes` — base64-inlined (MIME sniffed from magic bytes, else JPEG);
- a PIL `Image` — encoded to PNG and inlined.

`detail` (`"low"` | `"high"` | `"auto"`) is forwarded when set; the
block is the OpenAI/LiteLLM multimodal shape understood across providers.

```pycon
>>> to_image_content("https://x/y.jpg")
{'type': 'image_url', 'image_url': {'url': 'https://x/y.jpg'}}
>>> to_image_content("data:image/png;base64,AAAA")["image_url"]["url"][:10]
'data:image'
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
