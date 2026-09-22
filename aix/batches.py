"""Batch processing interface for AIX.

Efficiently process multiple prompts or embeddings in batches.

An item whose call raises does not stop the batch: its slot holds a
:class:`BatchError` -- a ``str`` subclass equal to the ``"ERROR: <message>"``
text these functions have always produced, but carrying the real exception so a
failure can be told apart from a model reply that happens to start "ERROR:".
Pass ``on_error='raise'`` or ``on_error='skip'`` to opt out of that slot.

Examples:
    Batch chat:
    >>> from aix.batches import batch_chat
    >>> prompts = ["What is 2+2?", "What is 3+3?", "What is 4+4?"]
    >>> results = list(batch_chat(prompts))  # doctest: +SKIP
    >>> len(results)  # doctest: +SKIP
    3

    Batch embeddings:
    >>> from aix.batches import batch_embeddings
    >>> texts = ["hello", "world", "foo", "bar"]
    >>> vectors = list(batch_embeddings(texts))  # doctest: +SKIP
    >>> len(vectors)  # doctest: +SKIP
    4
"""

from collections.abc import Iterable, Sequence
from typing import Union, Any, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Import from aix modules
from aix.chat import chat, _normalize_prompt
from aix.embeddings import embeddings

# Default configurations
DFLT_BATCH_SIZE = 10
DFLT_MAX_WORKERS = 5
DFLT_RETRY_ATTEMPTS = 3
DFLT_RETRY_DELAY = 1.0

# What a batch function does with an item that raised.
OnError = Literal["return", "raise", "skip"]
_ON_ERROR_CHOICES = ("return", "raise", "skip")


class BatchError(str):
    """A batch result slot standing in for an item that raised.

    Subclasses ``str`` so it *is* the ``"ERROR: ..."`` string these functions
    have always put in the failed slot -- comparisons, ``.startswith``,
    slicing, JSON serialisation and logging all behave identically. What it
    adds is the ability to tell a failure apart from a model that legitimately
    replied with text starting "ERROR:", and to reach the original exception.

    Attributes:
        exception: The exception that was caught.
        index: Position of the failed item in the input, when known.

    Examples:
        >>> err = BatchError(RuntimeError("rate limit exceeded"), index=1)
        >>> err == "ERROR: rate limit exceeded"
        True
        >>> isinstance(err, str), isinstance(err.exception, RuntimeError)
        (True, True)
        >>> err.index
        1
    """

    exception: BaseException
    index: Union[int, None]

    def __new__(
        cls,
        exception: BaseException,
        *,
        index: int = None,
        message: str = None,
    ):
        """Build the slot value.

        Args:
            exception: The exception that was caught.
            index: Position of the failed item in the input.
            message: The string value, when it must be restored verbatim
                (pickle round-trips). Defaults to the historical text.
        """
        obj = super().__new__(
            cls, f"ERROR: {exception}" if message is None else message
        )
        obj.exception = exception
        obj.index = index
        return obj

    def __getnewargs_ex__(self):
        """Keep the exception, index and exact text across a pickle round-trip.

        Without this, ``str.__getnewargs__`` would feed the *text* back into
        ``__new__`` as the exception, doubling the "ERROR: " prefix.
        """
        return (self.exception,), {"index": self.index, "message": str(self)}

    def __repr__(self):
        return f"{type(self).__name__}({str(self)!r}, index={self.index!r})"


def _check_on_error(on_error: str) -> None:
    """Reject an unsupported ``on_error`` value instead of ignoring it.

    Examples:
        >>> _check_on_error("skip")
        >>> _check_on_error("ignore")
        Traceback (most recent call last):
            ...
        ValueError: on_error must be one of ('return', 'raise', 'skip'), got 'ignore'
    """
    if on_error not in _ON_ERROR_CHOICES:
        raise ValueError(
            f"on_error must be one of {_ON_ERROR_CHOICES}, got {on_error!r}"
        )


def _chunk_iterable(iterable: Iterable, chunk_size: int) -> Iterable[list]:
    """Split an iterable into chunks.

    Args:
        iterable: Input iterable
        chunk_size: Size of each chunk

    Yields:
        Lists of items, each up to chunk_size length

    Examples:
        >>> list(_chunk_iterable(range(10), 3))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    """
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def batch_chat(
    prompts: Iterable[Union[str, list[dict]]],
    *,
    model: str = None,
    batch_size: int = None,
    max_workers: int = None,
    show_progress: bool = False,
    on_error: OnError = "return",
    **chat_kwargs,
) -> Iterable[str]:
    """Process multiple chat prompts in batches.

    This function processes multiple prompts efficiently by:
    1. Chunking prompts into batches
    2. Processing batches in parallel where possible
    3. Yielding results in the same order as input

    Args:
        prompts: Iterable of prompts (strings or message lists)
        model: Model to use for all prompts
        batch_size: Number of prompts to process in each batch
        max_workers: Maximum number of parallel workers
        show_progress: If True, print progress information
        on_error: What to do with a prompt whose call raised.
            `'return'` (default) yields a `BatchError` in that slot, keeping
            the stream the same length and order as the input.
            `'raise'` stops after the batch in which the first failure landed
            and re-raises the lowest-indexed exception of that batch.
            `'skip'` omits the failed slots from the stream.
        **chat_kwargs: Additional parameters passed to chat()

    Yields:
        Responses in the same order as input prompts. A prompt that raised
        yields a `BatchError` -- a `str` subclass equal to `"ERROR: <message>"`,
        carrying the original exception on `.exception`.

    Raises:
        ValueError: If `on_error` is not one of 'return', 'raise', 'skip'.

    Examples:
        >>> from aix.batches import batch_chat
        >>> prompts = [
        ...     "What is 2+2?",
        ...     "What is 3+3?",
        ...     "What is 5+5?"
        ... ]
        >>> results = list(batch_chat(prompts))  # doctest: +SKIP
        >>> len(results)  # doctest: +SKIP
        3

        >>> # With specific model
        >>> results = list(batch_chat(
        ...     prompts,
        ...     model="gpt-4o-mini",
        ...     batch_size=5
        ... ))  # doctest: +SKIP

        >>> # Process large dataset
        >>> def generate_prompts():
        ...     for i in range(100):
        ...         yield f"Explain concept {i}"
        >>> results = batch_chat(
        ...     generate_prompts(),
        ...     show_progress=True
        ... )  # doctest: +SKIP
        >>> for i, result in enumerate(results):  # doctest: +SKIP
        ...     print(f"Result {i}: {result[:50]}...")

        >>> # Tell a failure apart from a reply that starts with "ERROR:"
        >>> for result in batch_chat(prompts):  # doctest: +SKIP
        ...     if isinstance(result, BatchError):
        ...         raise result.exception
    """
    _check_on_error(on_error)

    batch_size = batch_size or DFLT_BATCH_SIZE
    max_workers = max_workers or DFLT_MAX_WORKERS

    # Convert to list to allow indexing
    prompts_list = list(prompts)
    total = len(prompts_list)

    if show_progress:
        print(f"Processing {total} prompts in batches of {batch_size}...")

    results = [None] * total
    errors = {}
    processed = 0

    # Process in chunks with parallel execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for chunk_idx, chunk in enumerate(_chunk_iterable(prompts_list, batch_size)):
            # Submit all prompts in this chunk
            futures = {}
            for i, prompt in enumerate(chunk):
                idx = chunk_idx * batch_size + i
                future = executor.submit(chat, prompt, model=model, **chat_kwargs)
                futures[future] = idx

            # Collect results as they complete
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results[idx] = result
                    processed += 1

                    if show_progress and processed % 10 == 0:
                        print(f"Processed {processed}/{total} prompts")

                except Exception as e:
                    # Store error as result
                    errors[idx] = results[idx] = BatchError(e, index=idx)
                    if show_progress:
                        print(f"Error processing prompt {idx}: {e}")

            if errors and on_error == "raise":
                # Don't spend the rest of the budget on a batch already lost.
                break

    if errors and on_error == "raise":
        raise errors[min(errors)].exception

    if show_progress:
        print(f"Completed processing {total} prompts")

    # Yield results in order
    for idx, result in enumerate(results):
        if on_error == "skip" and idx in errors:
            continue
        yield result


def batch_embeddings(
    segments: Iterable[str],
    *,
    model: str = None,
    batch_size: int = None,
    show_progress: bool = False,
    **embedding_kwargs,
) -> Iterable[Sequence[float]]:
    """Generate embeddings for multiple texts in batches.

    For efficiency, this function processes embeddings in chunks,
    as most embedding APIs can handle multiple texts per request.

    Args:
        segments: Iterable of text strings to embed
        model: Embedding model to use
        batch_size: Number of texts per batch
        show_progress: If True, print progress information
        **embedding_kwargs: Additional parameters for embeddings()

    Yields:
        Vector embeddings in the same order as input

    Examples:
        >>> from aix.batches import batch_embeddings
        >>> texts = ["hello", "world", "foo", "bar"] * 25  # 100 texts
        >>> vectors = list(batch_embeddings(
        ...     texts,
        ...     batch_size=10,
        ...     show_progress=True
        ... ))  # doctest: +SKIP
        >>> len(vectors)  # doctest: +SKIP
        100

        >>> # Process large dataset efficiently
        >>> def read_documents():
        ...     # Generator that yields documents
        ...     for i in range(1000):
        ...         yield f"Document {i} content"
        >>> all_vectors = []
        >>> for vec in batch_embeddings(read_documents()):  # doctest: +SKIP
        ...     all_vectors.append(vec)
    """
    batch_size = batch_size or DFLT_BATCH_SIZE

    segments_list = list(segments)
    total = len(segments_list)

    if show_progress:
        print(f"Generating embeddings for {total} texts in batches of {batch_size}...")

    processed = 0

    # Process in chunks
    for chunk in _chunk_iterable(segments_list, batch_size):
        # Get embeddings for this chunk
        chunk_embeddings = list(embeddings(chunk, model=model, **embedding_kwargs))

        processed += len(chunk)
        if show_progress and processed % 100 == 0:
            print(f"Generated {processed}/{total} embeddings")

        # Yield results
        for vec in chunk_embeddings:
            yield vec

    if show_progress:
        print(f"Completed {total} embeddings")


def batch_process(
    items: Iterable[Any],
    process_func: callable,
    *,
    batch_size: int = None,
    max_workers: int = None,
    show_progress: bool = False,
    retry_attempts: int = None,
    retry_delay: float = None,
    on_error: OnError = "return",
) -> Iterable[Any]:
    """Generic batch processing with parallel execution and retries.

    This is a general-purpose batch processor that can be used for
    any operation, not just chat or embeddings.

    Args:
        items: Items to process
        process_func: Function to apply to each item
        batch_size: Batch size for chunking
        max_workers: Maximum parallel workers
        show_progress: Show progress information
        retry_attempts: Number of retry attempts on failure
        retry_delay: Delay between retries (seconds)
        on_error: What to do with an item still failing after every retry.
            `'return'` (default) yields a `BatchError` in that slot, `'raise'`
            re-raises, `'skip'` omits it. See :func:`batch_chat`.

    Yields:
        Results in same order as input. An item that raised yields a
        `BatchError` -- a `str` subclass equal to `"ERROR: <message>"`,
        carrying the original exception on `.exception`.

    Raises:
        ValueError: If `on_error` is not one of 'return', 'raise', 'skip'.

    Examples:
        >>> from aix.batches import batch_process
        >>> from aix.chat import chat

        >>> # Custom processing function
        >>> def analyze_sentiment(text):
        ...     return chat(f"Analyze sentiment: {text}")

        >>> texts = ["I love this!", "This is terrible", "It's okay"]
        >>> results = list(batch_process(
        ...     texts,
        ...     analyze_sentiment,
        ...     batch_size=5
        ... ))  # doctest: +SKIP

        >>> # With retries for flaky operations
        >>> def flaky_api_call(item):
        ...     # Some API that might fail
        ...     return call_api(item)

        >>> results = batch_process(
        ...     items,
        ...     flaky_api_call,
        ...     retry_attempts=3,
        ...     retry_delay=2.0
        ... )  # doctest: +SKIP
    """
    _check_on_error(on_error)

    batch_size = batch_size or DFLT_BATCH_SIZE
    max_workers = max_workers or DFLT_MAX_WORKERS
    retry_attempts = retry_attempts or DFLT_RETRY_ATTEMPTS
    retry_delay = retry_delay or DFLT_RETRY_DELAY

    items_list = list(items)
    total = len(items_list)

    if show_progress:
        print(f"Processing {total} items with {max_workers} workers...")

    results = [None] * total
    errors = {}
    processed = 0

    def process_with_retry(item, idx):
        """Process item with retry logic."""
        for attempt in range(retry_attempts):
            try:
                return process_func(item)
            except Exception as e:
                if attempt < retry_attempts - 1:
                    if show_progress:
                        print(
                            f"Retry {attempt + 1}/{retry_attempts} for item {idx}: {e}"
                        )
                    time.sleep(retry_delay * (2**attempt))  # Exponential backoff
                else:
                    raise

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for chunk_idx, chunk in enumerate(_chunk_iterable(items_list, batch_size)):
            futures = {}
            for i, item in enumerate(chunk):
                idx = chunk_idx * batch_size + i
                future = executor.submit(process_with_retry, item, idx)
                futures[future] = idx

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results[idx] = result
                    processed += 1

                    if show_progress and processed % 10 == 0:
                        print(f"Processed {processed}/{total} items")

                except Exception as e:
                    errors[idx] = results[idx] = BatchError(e, index=idx)
                    if show_progress:
                        print(f"Error processing item {idx}: {e}")

            if errors and on_error == "raise":
                # Don't spend the rest of the budget on a batch already lost.
                break

    if errors and on_error == "raise":
        raise errors[min(errors)].exception

    if show_progress:
        print(f"Completed {total} items")

    for idx, result in enumerate(results):
        if on_error == "skip" and idx in errors:
            continue
        yield result


class BatchProcessor:
    """Stateful batch processor for managing long-running operations.

    Provides a higher-level interface for batch processing with
    progress tracking, error handling, and result caching.

    Examples:
        >>> processor = BatchProcessor(show_progress=True)  # doctest: +SKIP
        >>> results = processor.process_chats(prompts)  # doctest: +SKIP
        >>> processor.save_results("output.json")  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        batch_size: int = None,
        max_workers: int = None,
        show_progress: bool = True,
    ):
        """Initialize batch processor.

        Args:
            batch_size: Default batch size
            max_workers: Default max workers
            show_progress: Show progress by default
        """
        self.batch_size = batch_size or DFLT_BATCH_SIZE
        self.max_workers = max_workers or DFLT_MAX_WORKERS
        self.show_progress = show_progress
        self.results = []
        self.errors = []

    def process_chats(
        self, prompts: Iterable[Union[str, list[dict]]], **kwargs
    ) -> list[str]:
        """Process chat prompts and store results.

        Args:
            prompts: Prompts to process
            **kwargs: Additional parameters for batch_chat()

        Returns:
            List of results
        """
        results = list(
            batch_chat(
                prompts,
                batch_size=self.batch_size,
                max_workers=self.max_workers,
                show_progress=self.show_progress,
                **kwargs,
            )
        )
        self.results = results
        return results

    def process_embeddings(
        self, texts: Iterable[str], **kwargs
    ) -> list[Sequence[float]]:
        """Process embeddings and store results.

        Args:
            texts: Texts to embed
            **kwargs: Additional parameters for batch_embeddings()

        Returns:
            List of embedding vectors
        """
        results = list(
            batch_embeddings(
                texts,
                batch_size=self.batch_size,
                show_progress=self.show_progress,
                **kwargs,
            )
        )
        self.results = results
        return results

    def save_results(self, filepath: str):
        """Save results to file.

        Args:
            filepath: Path to save results (JSON)
        """
        import json
        from pathlib import Path

        path = Path(filepath)
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)

    def clear(self):
        """Clear stored results and errors."""
        self.results = []
        self.errors = []
