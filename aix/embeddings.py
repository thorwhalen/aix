"""Embeddings interface for AIX.

Generate vector embeddings from text using various models and providers.

Examples:
    Simple embedding:
    >>> from aix.embeddings import embeddings
    >>> vecs = list(embeddings(["hello", "world"]))  # doctest: +SKIP
    >>> len(vecs)  # doctest: +SKIP
    2
    >>> len(vecs[0])  # dimension depends on model  # doctest: +SKIP
    1536

    With specific model:
    >>> vecs = list(embeddings(
    ...     ["hello", "world"],
    ...     model="text-embedding-3-small"
    ... ))  # doctest: +SKIP

    Single text:
    >>> vec = embed("hello world")  # doctest: +SKIP
    >>> len(vec)  # doctest: +SKIP
    1536
"""

from collections.abc import Iterable, Sequence
from typing import Union

# Import LiteLLM but keep it private
try:
    from litellm import embedding as _litellm_embedding
except ImportError:
    _litellm_embedding = None


# Default configurations
DFLT_EMBEDDING_MODEL = 'text-embedding-3-small'


def embeddings(
    segments: Iterable[str],
    *,
    model: str = None,
    **kwargs
) -> Iterable[Sequence[float]]:
    """Generate embeddings for multiple text segments.

    This is the main embedding interface for AIX. It abstracts away provider-specific
    details and provides a clean, consistent API across all embedding models.

    Args:
        segments: Iterable of text strings to embed
        model: Model identifier (e.g., 'text-embedding-3-small', 'text-embedding-ada-002',
            'openrouter/openai/text-embedding-3-small'). If None, uses default.
        **kwargs: Additional provider-specific parameters passed to LiteLLM

    Yields:
        Vector embeddings as sequences of floats. Each embedding corresponds to
        one input segment in the same order.

    Raises:
        ImportError: If LiteLLM is not installed
        ValueError: If segments is empty or invalid

    Examples:
        >>> from aix.embeddings import embeddings
        >>> texts = ["cat", "dog", "bird"]
        >>> vecs = list(embeddings(texts))  # doctest: +SKIP
        >>> len(vecs)  # doctest: +SKIP
        3

        >>> # With specific model
        >>> vecs = list(embeddings(
        ...     ["hello", "world"],
        ...     model="text-embedding-3-large"
        ... ))  # doctest: +SKIP

        >>> # Process in chunks for large datasets
        >>> def chunk_texts(texts, size=100):
        ...     for i in range(0, len(texts), size):
        ...         yield texts[i:i+size]
        >>> all_vecs = []
        >>> for chunk in chunk_texts(large_dataset):  # doctest: +SKIP
        ...     all_vecs.extend(embeddings(chunk))
    """
    if _litellm_embedding is None:
        raise ImportError(
            "LiteLLM is required for embeddings functionality. "
            "Install it with: pip install litellm"
        )

    # Convert to list to allow validation
    segments_list = list(segments)

    if not segments_list:
        raise ValueError("Cannot generate embeddings for empty sequence")

    # Apply defaults
    model = model or DFLT_EMBEDDING_MODEL

    # Build LiteLLM parameters
    litellm_kwargs = {
        'model': model,
        'input': segments_list,
    }

    # Add any additional provider-specific kwargs
    litellm_kwargs.update(kwargs)

    # Call LiteLLM
    response = _litellm_embedding(**litellm_kwargs)

    # Extract embeddings from response
    # LiteLLM returns a response with .data list
    # Each item has .embedding attribute
    for item in response.data:
        yield item['embedding']


def embed(
    text: str,
    *,
    model: str = None,
    **kwargs
) -> Sequence[float]:
    """Generate embedding for a single text.

    Convenience function for embedding a single text string.

    Args:
        text: Text string to embed
        model: Model identifier
        **kwargs: Additional parameters for embeddings()

    Returns:
        Vector embedding as sequence of floats

    Examples:
        >>> from aix.embeddings import embed
        >>> vec = embed("Hello, world!")  # doctest: +SKIP
        >>> len(vec)  # doctest: +SKIP
        1536

        >>> # Compare similarity
        >>> import numpy as np
        >>> v1 = np.array(embed("cat"))  # doctest: +SKIP
        >>> v2 = np.array(embed("kitten"))  # doctest: +SKIP
        >>> v3 = np.array(embed("computer"))  # doctest: +SKIP
        >>> # Cosine similarity
        >>> np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))  # doctest: +SKIP
        0.92  # High similarity
        >>> np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3))  # doctest: +SKIP
        0.23  # Low similarity
    """
    return next(embeddings([text], model=model, **kwargs))


class EmbeddingCache:
    """Cache for embeddings to avoid redundant API calls.

    Useful when you need to embed the same texts multiple times.

    Examples:
        >>> cache = EmbeddingCache()  # doctest: +SKIP
        >>> vec1 = cache.embed("hello")  # API call  # doctest: +SKIP
        >>> vec2 = cache.embed("hello")  # From cache  # doctest: +SKIP
        >>> vec1 == vec2  # doctest: +SKIP
        True
    """

    def __init__(self, model: str = None, **embedding_kwargs):
        """Initialize cache.

        Args:
            model: Default model for embeddings
            **embedding_kwargs: Additional parameters for embeddings()
        """
        self._cache = {}
        self._model = model
        self._embedding_kwargs = embedding_kwargs

    def embed(self, text: str, force_refresh: bool = False) -> Sequence[float]:
        """Get embedding for text, using cache if available.

        Args:
            text: Text to embed
            force_refresh: If True, bypass cache and get fresh embedding

        Returns:
            Vector embedding
        """
        cache_key = (text, self._model)

        if not force_refresh and cache_key in self._cache:
            return self._cache[cache_key]

        # Get fresh embedding
        vec = embed(text, model=self._model, **self._embedding_kwargs)
        self._cache[cache_key] = vec
        return vec

    def embed_batch(
        self,
        texts: Iterable[str],
        force_refresh: bool = False
    ) -> list[Sequence[float]]:
        """Get embeddings for multiple texts, using cache when possible.

        Args:
            texts: Texts to embed
            force_refresh: If True, bypass cache

        Returns:
            List of embeddings in same order as input texts
        """
        texts_list = list(texts)
        results = []
        uncached_texts = []
        uncached_indices = []

        # Check cache
        for i, text in enumerate(texts_list):
            cache_key = (text, self._model)
            if not force_refresh and cache_key in self._cache:
                results.append(self._cache[cache_key])
            else:
                results.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Get fresh embeddings for uncached texts
        if uncached_texts:
            new_embeddings = list(
                embeddings(uncached_texts, model=self._model, **self._embedding_kwargs)
            )

            # Update cache and results
            for text, vec, idx in zip(uncached_texts, new_embeddings, uncached_indices):
                cache_key = (text, self._model)
                self._cache[cache_key] = vec
                results[idx] = vec

        return results

    def clear(self):
        """Clear the cache."""
        self._cache.clear()

    def __len__(self) -> int:
        """Return number of cached embeddings."""
        return len(self._cache)


def cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity (between -1 and 1)

    Examples:
        >>> from aix.embeddings import embed, cosine_similarity
        >>> v1 = embed("cat")  # doctest: +SKIP
        >>> v2 = embed("kitten")  # doctest: +SKIP
        >>> similarity = cosine_similarity(v1, v2)  # doctest: +SKIP
        >>> similarity > 0.8  # doctest: +SKIP
        True
    """
    # Compute dot product
    dot_product = sum(a * b for a, b in zip(vec1, vec2))

    # Compute magnitudes
    mag1 = sum(a * a for a in vec1) ** 0.5
    mag2 = sum(b * b for b in vec2) ** 0.5

    # Avoid division by zero
    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot_product / (mag1 * mag2)


def find_most_similar(
    query: Union[str, Sequence[float]],
    candidates: Iterable[str],
    *,
    model: str = None,
    top_k: int = 5,
    **kwargs
) -> list[tuple[str, float]]:
    """Find most similar texts to a query.

    Args:
        query: Query text or pre-computed embedding vector
        candidates: Candidate texts to compare against
        model: Embedding model to use
        top_k: Number of top results to return
        **kwargs: Additional parameters for embeddings()

    Returns:
        List of (text, similarity_score) tuples, sorted by similarity (highest first)

    Examples:
        >>> query = "What is machine learning?"
        >>> docs = [
        ...     "Machine learning is a type of AI",
        ...     "Python is a programming language",
        ...     "Neural networks are used in deep learning",
        ... ]
        >>> results = find_most_similar(query, docs, top_k=2)  # doctest: +SKIP
        >>> results[0][0]  # Most similar doc  # doctest: +SKIP
        'Machine learning is a type of AI'
    """
    # Get query embedding if needed
    if isinstance(query, str):
        query_vec = embed(query, model=model, **kwargs)
    else:
        query_vec = query

    # Get candidate embeddings
    candidates_list = list(candidates)
    candidate_vecs = list(embeddings(candidates_list, model=model, **kwargs))

    # Compute similarities
    similarities = [
        (text, cosine_similarity(query_vec, vec))
        for text, vec in zip(candidates_list, candidate_vecs)
    ]

    # Sort by similarity (descending) and return top_k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]
