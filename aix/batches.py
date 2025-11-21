"""Batch processing interface for AIX.

Efficiently process multiple prompts or embeddings in batches.

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
from typing import Union, Any
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
    **chat_kwargs
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
        **chat_kwargs: Additional parameters passed to chat()

    Yields:
        Responses in the same order as input prompts

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
    """
    batch_size = batch_size or DFLT_BATCH_SIZE
    max_workers = max_workers or DFLT_MAX_WORKERS

    # Convert to list to allow indexing
    prompts_list = list(prompts)
    total = len(prompts_list)

    if show_progress:
        print(f"Processing {total} prompts in batches of {batch_size}...")

    results = [None] * total
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
                    results[idx] = f"ERROR: {str(e)}"
                    if show_progress:
                        print(f"Error processing prompt {idx}: {e}")

    if show_progress:
        print(f"Completed processing {total} prompts")

    # Yield results in order
    for result in results:
        yield result


def batch_embeddings(
    segments: Iterable[str],
    *,
    model: str = None,
    batch_size: int = None,
    show_progress: bool = False,
    **embedding_kwargs
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
    retry_delay: float = None
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

    Yields:
        Results in same order as input

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
    batch_size = batch_size or DFLT_BATCH_SIZE
    max_workers = max_workers or DFLT_MAX_WORKERS
    retry_attempts = retry_attempts or DFLT_RETRY_ATTEMPTS
    retry_delay = retry_delay or DFLT_RETRY_DELAY

    items_list = list(items)
    total = len(items_list)

    if show_progress:
        print(f"Processing {total} items with {max_workers} workers...")

    results = [None] * total
    processed = 0

    def process_with_retry(item, idx):
        """Process item with retry logic."""
        for attempt in range(retry_attempts):
            try:
                return process_func(item)
            except Exception as e:
                if attempt < retry_attempts - 1:
                    if show_progress:
                        print(f"Retry {attempt + 1}/{retry_attempts} for item {idx}: {e}")
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
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
                    results[idx] = f"ERROR: {str(e)}"
                    if show_progress:
                        print(f"Error processing item {idx}: {e}")

    if show_progress:
        print(f"Completed {total} items")

    for result in results:
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
        show_progress: bool = True
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
        self,
        prompts: Iterable[Union[str, list[dict]]],
        **kwargs
    ) -> list[str]:
        """Process chat prompts and store results.

        Args:
            prompts: Prompts to process
            **kwargs: Additional parameters for batch_chat()

        Returns:
            List of results
        """
        results = list(batch_chat(
            prompts,
            batch_size=self.batch_size,
            max_workers=self.max_workers,
            show_progress=self.show_progress,
            **kwargs
        ))
        self.results = results
        return results

    def process_embeddings(
        self,
        texts: Iterable[str],
        **kwargs
    ) -> list[Sequence[float]]:
        """Process embeddings and store results.

        Args:
            texts: Texts to embed
            **kwargs: Additional parameters for batch_embeddings()

        Returns:
            List of embedding vectors
        """
        results = list(batch_embeddings(
            texts,
            batch_size=self.batch_size,
            show_progress=self.show_progress,
            **kwargs
        ))
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
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)

    def clear(self):
        """Clear stored results and errors."""
        self.results = []
        self.errors = []
