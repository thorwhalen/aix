"""Unified model management interface.

Provides a high-level API for managing AI models across providers.
"""

from pathlib import Path
from typing import Any, Iterable, Callable

from aix.ai_models.base import Model, ModelRegistry, ConnectorRegistry, ModelSource
from aix.ai_models.sources import (
    Connector,
    OpenRouterSource,
    OllamaSource,
    ProviderAPISource,
    OpenAIConnector,
    OpenRouterConnector,
    LangChainConnector,
    OllamaConnector,
    DSPyConnector,
)


class ModelManager:
    """Unified interface for AI model management.

    Combines model registry, sources, and connectors into a single facade.

    >>> manager = ModelManager()
    >>> manager.register_connector(OpenAIConnector())
    >>> len(manager.connectors)
    1
    """

    def __init__(self, *, storage_path: Path | None = None):
        """Initialize model manager with optional persistent storage.

        Args:
            storage_path: Path to JSON file for persisting model registry
        """
        self._registry = ModelRegistry(storage_path=storage_path)
        self._connectors = ConnectorRegistry()
        self._sources: dict[str, ModelSource] = {}

    @property
    def models(self) -> ModelRegistry:
        """Access to model registry for direct manipulation.

        >>> manager = ModelManager()
        >>> manager.models["test"] = Model(id="test", provider="test")
        >>> "test" in manager.models
        True
        """
        return self._registry

    @property
    def connectors(self) -> ConnectorRegistry:
        """Access to connector registry."""
        return self._connectors

    def register_source(self, name: str, source: ModelSource) -> None:
        """Register a model discovery source.

        >>> manager = ModelManager()
        >>> source = OpenRouterSource()
        >>> manager.register_source("openrouter", source)
        >>> "openrouter" in manager.list_sources()
        True
        """
        self._sources[name] = source

    def register_connector(self, connector: Connector) -> None:
        """Register a model connector.

        >>> manager = ModelManager()
        >>> connector = OpenAIConnector()
        >>> manager.register_connector(connector)
        >>> connector.name in manager.connectors
        True
        """
        self._connectors[connector.name] = connector

    def list_sources(self) -> list[str]:
        """List registered source names."""
        return list(self._sources.keys())

    def discover_from_source(
        self, source_name: str, *, auto_register: bool = True, verbose: bool = True
    ) -> list[Model]:
        """Discover models from a registered source.

        Args:
            source_name: Name of registered source
            auto_register: If True, automatically add discovered models to registry

        Returns:
            List of discovered models
        """
        if source_name not in self._sources:
            raise ValueError(f"Source '{source_name}' not registered")

        source = self._sources[source_name]
        if verbose:
            print(f"Discovering models from {source_name}...")
        discovered = list(source.discover_models())
        if verbose:
            print(f"  -> Found {len(discovered)} models")

        if auto_register:
            for model in discovered:
                self._registry[model.id] = model

        return discovered

    def discover_all(self, *, auto_register: bool = True) -> dict[str, list[Model]]:
        """Discover models from all registered sources.

        Returns:
            Dict mapping source name to list of discovered models
        """
        results = {}
        for source_name in self._sources:
            results[source_name] = self.discover_from_source(
                source_name, auto_register=auto_register
            )
        return results

    def list_models(
        self,
        *,
        provider: str | None = None,
        is_local: bool | None = None,
        min_context_size: int | None = None,
        max_context_size: int | None = None,
        has_capabilities: Iterable[str] | None = None,
        tags: Iterable[str] | None = None,
        custom_filter: Callable[[Model], bool] | None = None,
    ) -> list[Model]:
        """List models with optional filtering.

        >>> manager = ModelManager()
        >>> manager.models["gpt-4"] = Model(id="gpt-4", provider="openai")
        >>> manager.models["llama2"] = Model(id="llama2", provider="ollama", is_local=True)
        >>> local = manager.list_models(is_local=True)
        >>> len(local)
        1
        >>> local[0].id
        'llama2'
        """
        return self._registry.filter(
            provider=provider,
            is_local=is_local,
            min_context_size=min_context_size,
            max_context_size=max_context_size,
            has_capabilities=has_capabilities,
            tags=tags,
            custom_filter=custom_filter,
        )

    def get_model(self, model_id: str) -> Model:
        """Get a specific model by ID.

        >>> manager = ModelManager()
        >>> manager.models["gpt-4"] = Model(id="gpt-4", provider="openai")
        >>> model = manager.get_model("gpt-4")
        >>> model.id
        'gpt-4'
        """
        return self._registry[model_id]

    def get_connector_metadata(
        self, model_id: str, connector_name: str
    ) -> dict[str, Any]:
        """Get formatted metadata for a specific connector.

        Args:
            model_id: ID of the model
            connector_name: Name of the connector to format for

        Returns:
            Dict with metadata formatted for the specified connector

        >>> manager = ModelManager()
        >>> manager.models["gpt-4"] = Model(id="gpt-4", provider="openai")
        >>> manager.register_connector(OpenAIConnector())
        >>> meta = manager.get_connector_metadata("gpt-4", "openai")
        >>> meta["model"]
        'gpt-4'
        """
        model = self._registry[model_id]
        connector = self._connectors[connector_name]
        return connector.format_metadata(model)

    def add_model_tag(self, model_id: str, *tags: str) -> None:
        """Add tags to a model.

        >>> manager = ModelManager()
        >>> manager.models["gpt-4"] = Model(id="gpt-4", provider="openai")
        >>> manager.add_model_tag("gpt-4", "fast", "expensive")
        >>> "fast" in manager.models["gpt-4"].tags
        True
        """
        model = self._registry[model_id]
        model.tags.update(tags)
        # Trigger save if using persistent storage
        self._registry[model_id] = model

    def set_custom_metadata(self, model_id: str, key: str, value: Any) -> None:
        """Set custom metadata on a model.

        >>> manager = ModelManager()
        >>> manager.models["gpt-4"] = Model(id="gpt-4", provider="openai")
        >>> manager.set_custom_metadata("gpt-4", "my_note", "Use for production")
        >>> manager.models["gpt-4"].custom_metadata["my_note"]
        'Use for production'
        """
        model = self._registry[model_id]
        model.custom_metadata[key] = value
        # Trigger save
        self._registry[model_id] = model

    def create_model_group(self, group_name: str, model_ids: Iterable[str]) -> None:
        """Create a named group of models using tags.

        >>> manager = ModelManager()
        >>> manager.models["gpt-4"] = Model(id="gpt-4", provider="openai")
        >>> manager.models["claude-3"] = Model(id="claude-3", provider="anthropic")
        >>> manager.create_model_group("premium", ["gpt-4", "claude-3"])
        >>> premium = manager.list_models(tags=["group:premium"])
        >>> len(premium)
        2
        """
        group_tag = f"group:{group_name}"
        for model_id in model_ids:
            self.add_model_tag(model_id, group_tag)

    def list_model_groups(self) -> set[str]:
        """List all model groups.

        Returns:
            Set of group names (without 'group:' prefix)
        """
        groups = set()
        for model in self._registry.values():
            for tag in model.tags:
                if tag.startswith("group:"):
                    groups.add(tag[6:])  # Remove 'group:' prefix
        return groups


def _create_default_manager(storage_path: Path | None = None) -> ModelManager:
    """Create a manager with common sources and connectors pre-registered.

    Helper function to set up a fully configured manager.
    """
    manager = ModelManager(storage_path=storage_path)

    # Register common connectors
    manager.register_connector(OpenAIConnector())
    manager.register_connector(OpenRouterConnector())
    manager.register_connector(LangChainConnector())
    manager.register_connector(OllamaConnector())
    manager.register_connector(DSPyConnector())

    # Register common sources
    manager.register_source("openrouter", OpenRouterSource())
    manager.register_source("ollama", OllamaSource())

    return manager


# Convenience function for quick setup
def get_manager(storage_path: Path | str | None = None) -> ModelManager:
    """Get a pre-configured model manager.

    >>> manager = get_manager()
    >>> len(manager.connectors) > 0
    True
    """
    if isinstance(storage_path, str):
        storage_path = Path(storage_path)
    return _create_default_manager(storage_path)
