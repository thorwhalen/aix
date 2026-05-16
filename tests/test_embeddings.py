"""Tests for aix.embeddings module."""

import pytest
from unittest.mock import Mock, patch
from aix.embeddings import (
    embeddings,
    embed,
    cosine_similarity,
    find_most_similar,
    EmbeddingCache,
    batched_embeddings,
    cached_embeddings,
    iter_batches,
    text_cache_key,
)


class TestEmbeddings:
    """Tests for embeddings function."""

    @patch("aix.embeddings._litellm_embedding")
    def test_basic_embeddings(self, mock_embedding):
        """Test basic embeddings generation."""
        # Mock response
        mock_response = Mock()
        mock_response.data = [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]},
        ]
        mock_embedding.return_value = mock_response

        result = list(embeddings(["hello", "world"]))

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]

    @patch("aix.embeddings._litellm_embedding")
    def test_embeddings_with_model(self, mock_embedding):
        """Test embeddings with specific model."""
        mock_response = Mock()
        mock_response.data = [{"embedding": [0.1, 0.2]}]
        mock_embedding.return_value = mock_response

        list(embeddings(["test"], model="text-embedding-3-large"))

        call_kwargs = mock_embedding.call_args[1]
        assert call_kwargs["model"] == "text-embedding-3-large"

    def test_embeddings_empty_list(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="empty sequence"):
            list(embeddings([]))


class TestEmbed:
    """Tests for embed function."""

    @patch("aix.embeddings.embeddings")
    def test_embed_single_text(self, mock_embeddings):
        """Test embedding single text."""
        mock_embeddings.return_value = iter([[0.1, 0.2, 0.3]])

        result = embed("hello")

        assert result == [0.1, 0.2, 0.3]
        # Verify it was called with a list containing single item
        call_args = mock_embeddings.call_args[0][0]
        assert call_args == ["hello"]


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_identical_vectors(self):
        """Test similarity of identical vectors."""
        vec = [1.0, 2.0, 3.0]
        similarity = cosine_similarity(vec, vec)
        assert abs(similarity - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors."""
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 1e-6

    def test_opposite_vectors(self):
        """Test similarity of opposite vectors."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [-1.0, -2.0, -3.0]
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity - (-1.0)) < 1e-6

    def test_zero_vector(self):
        """Test that zero vector returns 0."""
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 2.0, 3.0]
        similarity = cosine_similarity(vec1, vec2)
        assert similarity == 0.0


class TestFindMostSimilar:
    """Tests for find_most_similar function."""

    @patch("aix.embeddings.embed")
    @patch("aix.embeddings.embeddings")
    def test_find_most_similar_with_string_query(self, mock_embeddings, mock_embed):
        """Test finding most similar with string query."""
        # Mock query embedding
        mock_embed.return_value = [1.0, 0.0, 0.0]

        # Mock candidate embeddings
        mock_embeddings.return_value = iter(
            [
                [1.0, 0.0, 0.0],  # similarity = 1.0
                [0.0, 1.0, 0.0],  # similarity = 0.0
                [0.5, 0.5, 0.0],  # similarity ≈ 0.707
            ]
        )

        candidates = ["doc1", "doc2", "doc3"]
        results = find_most_similar("query", candidates, top_k=2)

        assert len(results) == 2
        assert results[0][0] == "doc1"  # Most similar
        assert results[0][1] == 1.0
        assert results[1][0] == "doc3"  # Second most similar

    @patch("aix.embeddings.embeddings")
    def test_find_most_similar_with_vector_query(self, mock_embeddings):
        """Test finding most similar with pre-computed vector."""
        query_vec = [1.0, 0.0]

        mock_embeddings.return_value = iter(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )

        candidates = ["doc1", "doc2"]
        results = find_most_similar(query_vec, candidates)

        assert results[0][0] == "doc1"
        assert abs(results[0][1] - 1.0) < 1e-6

    @patch("aix.embeddings.embed")
    @patch("aix.embeddings.embeddings")
    def test_find_most_similar_top_k(self, mock_embeddings, mock_embed):
        """Test that top_k limits results."""
        mock_embed.return_value = [1.0, 0.0]
        mock_embeddings.return_value = iter(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.6, 0.4],
                [0.4, 0.6],
            ]
        )

        results = find_most_similar("query", ["a", "b", "c", "d"], top_k=2)

        assert len(results) == 2


class TestEmbeddingCache:
    """Tests for EmbeddingCache class."""

    @patch("aix.embeddings.embed")
    def test_cache_initialization(self, mock_embed):
        """Test cache initialization."""
        cache = EmbeddingCache()
        assert len(cache) == 0

    @patch("aix.embeddings.embed")
    def test_cache_stores_results(self, mock_embed):
        """Test that cache stores results."""
        mock_embed.return_value = [0.1, 0.2, 0.3]

        cache = EmbeddingCache()
        result1 = cache.embed("hello")

        assert len(cache) == 1
        assert result1 == [0.1, 0.2, 0.3]

    @patch("aix.embeddings.embed")
    def test_cache_reuses_results(self, mock_embed):
        """Test that cache reuses stored results."""
        mock_embed.return_value = [0.1, 0.2, 0.3]

        cache = EmbeddingCache()
        result1 = cache.embed("hello")
        result2 = cache.embed("hello")

        # Should only call embed once
        assert mock_embed.call_count == 1
        assert result1 == result2

    @patch("aix.embeddings.embed")
    def test_cache_force_refresh(self, mock_embed):
        """Test force refresh bypasses cache."""
        mock_embed.side_effect = [[0.1, 0.2], [0.3, 0.4]]

        cache = EmbeddingCache()
        result1 = cache.embed("hello")
        result2 = cache.embed("hello", force_refresh=True)

        # Should call embed twice
        assert mock_embed.call_count == 2
        assert result1 != result2

    @patch("aix.embeddings.embeddings")
    def test_cache_batch(self, mock_embeddings):
        """Test batch caching."""
        mock_embeddings.return_value = iter([[0.1, 0.2], [0.3, 0.4]])

        cache = EmbeddingCache()
        results = cache.embed_batch(["hello", "world"])

        assert len(results) == 2
        assert len(cache) == 2

    @patch("aix.embeddings.embed")
    @patch("aix.embeddings.embeddings")
    def test_cache_batch_partial_cache(self, mock_embeddings, mock_embed):
        """Test batch with partial cache hits."""
        # First, populate cache with one item
        mock_embed.return_value = [0.1, 0.2]
        cache = EmbeddingCache()
        cache.embed("hello")

        # Now batch request including cached and new items
        mock_embeddings.return_value = iter([[0.3, 0.4]])

        results = cache.embed_batch(["hello", "world"])

        # Should only fetch embedding for "world"
        mock_embeddings.assert_called_once()
        call_args = mock_embeddings.call_args[0][0]
        assert call_args == ["world"]

        assert len(results) == 2
        assert results[0] == [0.1, 0.2]  # From cache
        assert results[1] == [0.3, 0.4]  # New

    @patch("aix.embeddings.embed")
    def test_cache_clear(self, mock_embed):
        """Test clearing cache."""
        mock_embed.return_value = [0.1, 0.2]

        cache = EmbeddingCache()
        cache.embed("hello")
        assert len(cache) == 1

        cache.clear()
        assert len(cache) == 0


class TestIterBatches:
    """Tests for iter_batches helper."""

    def test_even_split(self):
        assert list(iter_batches([1, 2, 3, 4], size=2)) == [[1, 2], [3, 4]]

    def test_partial_tail(self):
        assert list(iter_batches([1, 2, 3, 4, 5], size=2)) == [[1, 2], [3, 4], [5]]

    def test_size_larger_than_input(self):
        assert list(iter_batches([1, 2], size=10)) == [[1, 2]]

    def test_empty(self):
        assert list(iter_batches([], size=3)) == []

    def test_invalid_size(self):
        with pytest.raises(ValueError):
            list(iter_batches([1], size=0))


class TestTextCacheKey:
    """Tests for text_cache_key."""

    def test_stable_for_same_inputs(self):
        assert text_cache_key("hello") == text_cache_key("hello")

    def test_model_affects_key(self):
        a = text_cache_key("hello", "text-embedding-3-small")
        b = text_cache_key("hello", "text-embedding-3-large")
        assert a != b

    def test_text_affects_key(self):
        assert text_cache_key("hello") != text_cache_key("world")

    def test_default_hash_len(self):
        assert len(text_cache_key("hello")) == 16

    def test_custom_hash_len(self):
        assert len(text_cache_key("hello", hash_len=8)) == 8

    def test_none_model_equals_default(self):
        from aix.embeddings import DFLT_EMBEDDING_MODEL

        assert text_cache_key("hello") == text_cache_key("hello", DFLT_EMBEDDING_MODEL)


class TestBatchedEmbeddings:
    """Tests for batched_embeddings."""

    @patch("aix.embeddings._litellm_embedding")
    def test_batches_correctly(self, mock_embedding):
        # Each call returns one embedding per input
        def side_effect(*, model, input, **kwargs):
            resp = Mock()
            resp.data = [{"embedding": [float(i)]} for i, _ in enumerate(input)]
            return resp

        mock_embedding.side_effect = side_effect
        vecs = list(batched_embeddings(["a", "b", "c", "d", "e"], batch_size=2))
        assert len(vecs) == 5
        # 3 API calls: 2, 2, 1
        assert mock_embedding.call_count == 3

    @patch("aix.embeddings._litellm_embedding")
    def test_on_batch_callback(self, mock_embedding):
        def side_effect(*, model, input, **kwargs):
            resp = Mock()
            resp.data = [{"embedding": [0.0]} for _ in input]
            return resp

        mock_embedding.side_effect = side_effect
        calls = []
        list(
            batched_embeddings(
                ["a", "b", "c"],
                batch_size=2,
                on_batch=lambda i, n: calls.append((i, n)),
            )
        )
        assert calls == [(0, 2), (1, 1)]


class TestCachedEmbeddings:
    """Tests for cached_embeddings (persistent caching pattern)."""

    @patch("aix.embeddings._litellm_embedding")
    def test_misses_then_hits(self, mock_embedding):
        def side_effect(*, model, input, **kwargs):
            resp = Mock()
            resp.data = [{"embedding": [float(len(s))]} for s in input]
            return resp

        mock_embedding.side_effect = side_effect

        cache: dict[str, list[float]] = {}
        vecs1 = cached_embeddings(["hi", "world"], cache=cache)
        # One API call, two cached entries
        assert mock_embedding.call_count == 1
        assert len(cache) == 2
        # Second call: pure hits
        mock_embedding.reset_mock()
        vecs2 = cached_embeddings(["hi", "world"], cache=cache)
        assert mock_embedding.call_count == 0
        assert vecs1 == vecs2

    @patch("aix.embeddings._litellm_embedding")
    def test_partial_hits(self, mock_embedding):
        # Pre-seed cache with one entry
        cache = {text_cache_key("hi"): [0.99]}

        def side_effect(*, model, input, **kwargs):
            resp = Mock()
            resp.data = [{"embedding": [float(len(s))]} for s in input]
            return resp

        mock_embedding.side_effect = side_effect

        vecs = cached_embeddings(["hi", "world"], cache=cache)
        # One API call for "world" only; "hi" was a hit
        assert mock_embedding.call_count == 1
        assert vecs[0] == [0.99]
        assert vecs[1] == [5.0]
        # Cache now has both
        assert len(cache) == 2

    @patch("aix.embeddings._litellm_embedding")
    def test_on_hit_fires_for_each_hit(self, mock_embedding):
        cache = {text_cache_key("hi"): [0.1]}

        def side_effect(*, model, input, **kwargs):
            resp = Mock()
            resp.data = [{"embedding": [0.0]} for _ in input]
            return resp

        mock_embedding.side_effect = side_effect

        hits: list[str] = []
        cached_embeddings(["hi", "hi", "miss"], cache=cache, on_hit=hits.append)
        # 'hi' appears twice -> 2 hits; 'miss' is a miss
        assert len(hits) == 2

    def test_empty_input(self):
        cache = {}
        assert cached_embeddings([], cache=cache) == []
