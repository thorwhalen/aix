"""Tests for aix.models module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from aix.models import (
    ModelStore,
    discover_available_models,
    get_model_info,
    find_models,
)
from aix.ai_models import Model


class TestModelStore:
    """Tests for ModelStore class."""

    @patch("aix.models.get_manager")
    def test_initialization(self, mock_get_manager):
        """Test ModelStore initialization."""
        mock_manager = Mock()
        mock_get_manager.return_value = mock_manager

        store = ModelStore()

        assert store._manager is mock_manager
        assert store._discovered is False

    @patch("aix.models.get_manager")
    def test_auto_discover(self, mock_get_manager):
        """Test auto_discover on initialization."""
        mock_manager = Mock()
        mock_manager.discover_from_source.return_value = []
        mock_get_manager.return_value = mock_manager

        store = ModelStore(auto_discover=True)

        mock_manager.discover_from_source.assert_called_once()

    @patch("aix.models.get_manager")
    def test_discover(self, mock_get_manager):
        """Test discover method."""
        mock_manager = Mock()
        mock_models = [
            Model(id="model1", provider="openai"),
            Model(id="model2", provider="openai"),
        ]
        mock_manager.discover_from_source.return_value = mock_models
        mock_get_manager.return_value = mock_manager

        store = ModelStore()
        result = store.discover("openrouter")

        assert result == mock_models
        assert store._discovered is True

    @patch("aix.models.get_manager")
    def test_getitem_by_id(self, mock_get_manager):
        """Test getting model by ID."""
        mock_manager = Mock()
        mock_model = Model(id="gpt-4", provider="openai")
        mock_manager.models = {"gpt-4": mock_model}
        mock_get_manager.return_value = mock_manager

        store = ModelStore()
        result = store["gpt-4"]

        assert result == mock_model

    @patch("aix.models.get_manager")
    def test_getitem_by_dict(self, mock_get_manager):
        """Test getting models by filter dict."""
        mock_manager = Mock()
        mock_models = [Model(id="gpt-4", provider="openai")]
        mock_manager.list_models.return_value = mock_models
        mock_get_manager.return_value = mock_manager

        store = ModelStore()
        result = store[{"provider": "openai"}]

        assert result == mock_models
        mock_manager.list_models.assert_called_once_with(provider="openai")

    @patch("aix.models.get_manager")
    def test_iter(self, mock_get_manager):
        """Test iterating over model IDs."""
        mock_manager = Mock()
        mock_manager.models = {"model1": Mock(), "model2": Mock()}
        mock_get_manager.return_value = mock_manager

        store = ModelStore()
        ids = list(store)

        assert "model1" in ids
        assert "model2" in ids

    @patch("aix.models.get_manager")
    def test_len(self, mock_get_manager):
        """Test getting number of models."""
        mock_manager = Mock()
        mock_manager.models = {"model1": Mock(), "model2": Mock()}
        mock_get_manager.return_value = mock_manager

        store = ModelStore()

        assert len(store) == 2

    @patch("aix.models.get_manager")
    def test_contains(self, mock_get_manager):
        """Test checking if model exists."""
        mock_manager = Mock()
        mock_manager.models = {"gpt-4": Mock()}
        mock_get_manager.return_value = mock_manager

        store = ModelStore()

        assert "gpt-4" in store
        assert "nonexistent" not in store

    @patch("aix.models.get_manager")
    def test_filter(self, mock_get_manager):
        """Test filtering models."""
        mock_manager = Mock()
        mock_registry = Mock()
        mock_filtered = [Model(id="gpt-4", provider="openai")]
        mock_registry.filter.return_value = mock_filtered
        mock_manager.models = mock_registry
        mock_get_manager.return_value = mock_manager

        store = ModelStore()
        result = store.filter(provider="openai")

        assert result == mock_filtered
        mock_registry.filter.assert_called_once_with(
            provider="openai",
            is_local=None,
            min_context_size=None,
            max_context_size=None,
            has_capabilities=None,
            tags=None,
            custom_filter=None,
        )

    @patch("aix.models.get_manager")
    def test_search(self, mock_get_manager):
        """Test searching models."""
        mock_manager = Mock()
        mock_models = {
            "openai/gpt-4": Model(id="openai/gpt-4", provider="openai", tags=set()),
            "openai/gpt-3.5": Model(id="openai/gpt-3.5", provider="openai", tags=set()),
            "anthropic/claude": Model(
                id="anthropic/claude", provider="anthropic", tags=set()
            ),
        }
        mock_manager.models.values.return_value = mock_models.values()
        mock_get_manager.return_value = mock_manager

        store = ModelStore()
        results = store.search("gpt")

        assert len(results) == 2
        assert all("gpt" in m.id.lower() for m in results)

    @patch("aix.models.get_manager")
    def test_by_provider(self, mock_get_manager):
        """Test getting models by provider."""
        mock_manager = Mock()
        mock_registry = Mock()
        mock_models = [Model(id="gpt-4", provider="openai")]
        mock_registry.filter.return_value = mock_models
        mock_manager.models = mock_registry
        mock_get_manager.return_value = mock_manager

        store = ModelStore()
        result = store.by_provider("openai")

        assert result == mock_models

    @patch("aix.models.get_manager")
    def test_by_task(self, mock_get_manager):
        """Test getting models by task."""
        mock_manager = Mock()
        mock_registry = Mock()
        mock_models = [Model(id="gpt-4", provider="openai")]
        mock_registry.filter.return_value = mock_models
        mock_manager.models = mock_registry
        mock_get_manager.return_value = mock_manager

        store = ModelStore()
        result = store.by_task("chat")

        mock_registry.filter.assert_called_once_with(
            provider=None,
            is_local=None,
            min_context_size=None,
            max_context_size=None,
            has_capabilities=["chat"],
            tags=None,
            custom_filter=None,
        )

    @patch("aix.models.get_manager")
    def test_recommend(self, mock_get_manager):
        """Test model recommendations."""
        mock_manager = Mock()
        mock_registry = Mock()

        # Create models with different costs
        model1 = Model(
            id="cheap",
            provider="openai",
            cost_per_token={"input": 0.0001, "output": 0.0002},
        )
        model2 = Model(
            id="expensive",
            provider="openai",
            cost_per_token={"input": 0.001, "output": 0.002},
        )
        mock_registry.filter.return_value = [model2, model1]  # Unsorted
        mock_manager.models = mock_registry
        mock_get_manager.return_value = mock_manager

        store = ModelStore()
        result = store.recommend(task="chat", max_cost_per_mtok=500.0)

        # Should return only model1 (cheaper) and sorted
        assert len(result) == 1
        assert result[0].id == "cheap"

    @patch("aix.models.get_manager")
    def test_get_info(self, mock_get_manager):
        """Test getting model info."""
        mock_manager = Mock()
        mock_model = Model(id="gpt-4", provider="openai")
        mock_manager.models = {"gpt-4": mock_model}
        mock_get_manager.return_value = mock_manager

        store = ModelStore()
        result = store.get_info("gpt-4")

        assert result == mock_model

    @patch("aix.models.get_manager")
    def test_get_connector_metadata(self, mock_get_manager):
        """Test getting connector metadata."""
        mock_manager = Mock()
        mock_metadata = {"model": "gpt-4", "temperature": 1.0}
        mock_manager.get_connector_metadata.return_value = mock_metadata
        mock_get_manager.return_value = mock_manager

        store = ModelStore()
        result = store.get_connector_metadata("gpt-4", "openai")

        assert result == mock_metadata


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @patch("aix.models.models")
    def test_discover_available_models(self, mock_models):
        """Test discover_available_models function."""
        mock_models.discover.return_value = []

        discover_available_models("openrouter")

        mock_models.discover.assert_called_once_with(source="openrouter", verbose=True)

    @patch("aix.models.models")
    def test_get_model_info(self, mock_models):
        """Test get_model_info function."""
        mock_model = Model(id="gpt-4", provider="openai")
        mock_models.get_info.return_value = mock_model

        result = get_model_info("gpt-4")

        assert result == mock_model
        mock_models.get_info.assert_called_once_with("gpt-4")

    @patch("aix.models.models")
    def test_find_models(self, mock_models):
        """Test find_models function."""
        mock_results = [Model(id="gpt-4", provider="openai")]
        mock_models.search.return_value = mock_results

        result = find_models("gpt")

        assert result == mock_results
        mock_models.search.assert_called_once_with("gpt")
