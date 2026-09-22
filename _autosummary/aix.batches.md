# aix.batches

Batch processing interface for AIX.

Efficiently process multiple prompts or embeddings in batches.

An item whose call raises does not stop the batch: its slot holds a
[`BatchError`](#aix.batches.BatchError) – a `str` subclass equal to the `"ERROR: <message>"`
text these functions have always produced, but carrying the real exception so a
failure can be told apart from a model reply that happens to start “ERROR:”.
Pass `on_error='raise'` or `on_error='skip'` to opt out of that slot.

### Examples

Batch chat:

```pycon
>>> from aix.batches import batch_chat
>>> prompts = ["What is 2+2?", "What is 3+3?", "What is 4+4?"]
>>> results = list(batch_chat(prompts))
>>> len(results)
3
```

Batch embeddings:

```pycon
>>> from aix.batches import batch_embeddings
>>> texts = ["hello", "world", "foo", "bar"]
>>> vectors = list(batch_embeddings(texts))
>>> len(vectors)
4
```

### Functions

| [`batch_chat`](#aix.batches.batch_chat)(prompts, \*[, model, batch_size, ...])   | Process multiple chat prompts in batches.                     |
|------------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| [`batch_embeddings`](#aix.batches.batch_embeddings)(segments, \*[, model, ...])        | Generate embeddings for multiple texts in batches.            |
| [`batch_process`](#aix.batches.batch_process)(items, process_func, \*[, ...])       | Generic batch processing with parallel execution and retries. |

### Classes

| [`BatchError`](#aix.batches.BatchError)(exception, \*[, index, message])        | A batch result slot standing in for an item that raised.       |
|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| [`BatchProcessor`](#aix.batches.BatchProcessor)(\*[, batch_size, max_workers, ...]) | Stateful batch processor for managing long-running operations. |

### *class* aix.batches.BatchError(exception, , index=None, message=None)

Bases: [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

A batch result slot standing in for an item that raised.

Subclasses `str` so it *is* the `"ERROR: ..."` string these functions
have always put in the failed slot – comparisons, `.startswith`,
slicing, JSON serialisation and logging all behave identically. What it
adds is the ability to tell a failure apart from a model that legitimately
replied with text starting “ERROR:”, and to reach the original exception.

#### exception

The exception that was caught.

#### index

Position of the failed item in the input, when known.

### Examples

```pycon
>>> err = BatchError(RuntimeError("rate limit exceeded"), index=1)
>>> err == "ERROR: rate limit exceeded"
True
>>> isinstance(err, str), isinstance(err.exception, RuntimeError)
(True, True)
>>> err.index
1
```

### *class* aix.batches.BatchProcessor(, batch_size=None, max_workers=None, show_progress=True)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Stateful batch processor for managing long-running operations.

Provides a higher-level interface for batch processing with
progress tracking, error handling, and result caching.

### Examples

```pycon
>>> processor = BatchProcessor(show_progress=True)
>>> results = processor.process_chats(prompts)
>>> processor.save_results("output.json")
```

#### clear()

Clear stored results and errors.

#### process_chats(prompts, \*\*kwargs)

Process chat prompts and store results.

* **Parameters:**
  * **prompts** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]]) – Prompts to process
  * **\*\*kwargs** – Additional parameters for batch_chat()
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]
* **Returns:**
  List of results

#### process_embeddings(texts, \*\*kwargs)

Process embeddings and store results.

* **Parameters:**
  * **texts** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Texts to embed
  * **\*\*kwargs** – Additional parameters for batch_embeddings()
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]
* **Returns:**
  List of embedding vectors

#### save_results(filepath)

Save results to file.

* **Parameters:**
  **filepath** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Path to save results (JSON)

### aix.batches.batch_chat(prompts, , model=None, batch_size=None, max_workers=None, show_progress=False, on_error='return', \*\*chat_kwargs)

Process multiple chat prompts in batches.

This function processes multiple prompts efficiently by:

1. Chunking prompts into batches
2. Processing batches in parallel where possible
3. Yielding results in the same order as input

* **Parameters:**
  * **prompts** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]]) – Iterable of prompts (strings or message lists)
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Model to use for all prompts
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of prompts to process in each batch
  * **max_workers** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum number of parallel workers
  * **show_progress** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, print progress information
  * **on_error** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'return'`, `'raise'`, `'skip'`]) – What to do with a prompt whose call raised.
    `'return'` (default) yields a `BatchError` in that slot, keeping
    the stream the same length and order as the input.
    `'raise'` stops after the batch in which the first failure landed
    and re-raises the lowest-indexed exception of that batch.
    `'skip'` omits the failed slots from the stream.
  * **\*\*chat_kwargs** – Additional parameters passed to chat()
* **Yields:**
  Responses in the same order as input prompts. A prompt that raised
  yields a `BatchError` – a `str` subclass equal to `"ERROR: <message>"`,
  carrying the original exception on `.exception`.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If `on_error` is not one of ‘return’, ‘raise’, ‘skip’.
* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### Examples

```pycon
>>> from aix.batches import batch_chat
>>> prompts = [
...     "What is 2+2?",
...     "What is 3+3?",
...     "What is 5+5?"
... ]
>>> results = list(batch_chat(prompts))
>>> len(results)
3
```

```pycon
>>> # With specific model
>>> results = list(batch_chat(
...     prompts,
...     model="gpt-4o-mini",
...     batch_size=5
... ))
```

```pycon
>>> # Process large dataset
>>> def generate_prompts():
...     for i in range(100):
...         yield f"Explain concept {i}"
>>> results = batch_chat(
...     generate_prompts(),
...     show_progress=True
... )
>>> for i, result in enumerate(results):
...     print(f"Result {i}: {result[:50]}...")
```

```pycon
>>> # Tell a failure apart from a reply that starts with "ERROR:"
>>> for result in batch_chat(prompts):
...     if isinstance(result, BatchError):
...         raise result.exception
```

### aix.batches.batch_embeddings(segments, , model=None, batch_size=None, show_progress=False, \*\*embedding_kwargs)

Generate embeddings for multiple texts in batches.

For efficiency, this function processes embeddings in chunks,
as most embedding APIs can handle multiple texts per request.

* **Parameters:**
  * **segments** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Iterable of text strings to embed
  * **model** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Embedding model to use
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of texts per batch
  * **show_progress** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, print progress information
  * **\*\*embedding_kwargs** – Additional parameters for embeddings()
* **Yields:**
  Vector embeddings in the same order as input
* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]

### Examples

```pycon
>>> from aix.batches import batch_embeddings
>>> texts = ["hello", "world", "foo", "bar"] * 25  # 100 texts
>>> vectors = list(batch_embeddings(
...     texts,
...     batch_size=10,
...     show_progress=True
... ))
>>> len(vectors)
100
```

```pycon
>>> # Process large dataset efficiently
>>> def read_documents():
...     # Generator that yields documents
...     for i in range(1000):
...         yield f"Document {i} content"
>>> all_vectors = []
>>> for vec in batch_embeddings(read_documents()):
...     all_vectors.append(vec)
```

### aix.batches.batch_process(items, process_func, , batch_size=None, max_workers=None, show_progress=False, retry_attempts=None, retry_delay=None, on_error='return')

Generic batch processing with parallel execution and retries.

This is a general-purpose batch processor that can be used for
any operation, not just chat or embeddings.

* **Parameters:**
  * **items** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – Items to process
  * **process_func** (`callable`) – Function to apply to each item
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Batch size for chunking
  * **max_workers** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum parallel workers
  * **show_progress** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Show progress information
  * **retry_attempts** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of retry attempts on failure
  * **retry_delay** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Delay between retries (seconds)
  * **on_error** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'return'`, `'raise'`, `'skip'`]) – What to do with an item still failing after every retry.
    `'return'` (default) yields a `BatchError` in that slot, `'raise'`
    re-raises, `'skip'` omits it. See [`batch_chat()`](#aix.batches.batch_chat).
* **Yields:**
  Results in same order as input. An item that raised yields a
  `BatchError` – a `str` subclass equal to `"ERROR: <message>"`,
  carrying the original exception on `.exception`.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If `on_error` is not one of ‘return’, ‘raise’, ‘skip’.
* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### Examples

```pycon
>>> from aix.batches import batch_process
>>> from aix.chat import chat
```

```pycon
>>> # Custom processing function
>>> def analyze_sentiment(text):
...     return chat(f"Analyze sentiment: {text}")
```

```pycon
>>> texts = ["I love this!", "This is terrible", "It's okay"]
>>> results = list(batch_process(
...     texts,
...     analyze_sentiment,
...     batch_size=5
... ))
```

```pycon
>>> # With retries for flaky operations
>>> def flaky_api_call(item):
...     # Some API that might fail
...     return call_api(item)
```

```pycon
>>> results = batch_process(
...     items,
...     flaky_api_call,
...     retry_attempts=3,
...     retry_delay=2.0
... )
```
