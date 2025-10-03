"""AI Model Management Module.

A unified interface for managing AI models across multiple providers.

Basic usage:
    >>> from aix.ai_models import get_manager
    >>> manager = get_manager()
    >>> _ = manager.discover_from_source("openrouter", auto_register=True, verbose=False)
    >>> models = manager.list_models(provider="openai")

Custom filtering:
    >>> cheap_models = manager.list_models(
    ...     custom_filter=lambda m: m.cost_per_token.get("input", 0) < 0.001
    ... )

Get connector-specific metadata:
    >>> openai_meta = manager.get_connector_metadata("openai/gpt-4", "openai")
    >>> # Use with: openai.ChatCompletion.create(**openai_meta, messages=[...])

"""

# Core types
from aix.ai_models.base import (
    Model,
    ModelRegistry,
    ModelSource,
    Connector,
    ConnectorRegistry,
)

# Concrete sources and connectors
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

# Main facade
from aix.ai_models.manager import (
    ModelManager,
    get_manager,
)

# Version info
__version__ = "0.1.0"

# Public API
__all__ = [
    # Core types
    "Model",
    "ModelRegistry",
    "ModelSource",
    "Connector",
    "ConnectorRegistry",
    # Sources
    "OpenRouterSource",
    "OllamaSource",
    "ProviderAPISource",
    # Connectors
    "OpenAIConnector",
    "OpenRouterConnector",
    "LangChainConnector",
    "OllamaConnector",
    "DSPyConnector",
    # Main API
    "ModelManager",
    "get_manager",
]


# Convenience functions for common operations

def list_available_models(
    *,
    provider: str | None = None,
    is_local: bool | None = None,
    storage_path: str | None = None
) -> list[Model]:
    """Quick function to list available models without explicit manager.
    
    >>> models = list_available_models(provider="openai")
    >>> len(models) >= 0
    True
    """
    manager = get_manager(storage_path=storage_path)
    return manager.list_models(provider=provider, is_local=is_local)


def discover_models(
    source_name: str = "openrouter",
    *,
    storage_path: str | None = None,
    auto_register: bool = True,
    verbose: bool = True,
) -> list[Model]:
    """Quick function to discover models from a source.
    
    >>> models = discover_models("openrouter", auto_register=False, verbose=False)
    >>> len(models) > 0
    True
    """
    manager = get_manager(storage_path=storage_path)
    return manager.discover_from_source(
        source_name, auto_register=auto_register, verbose=verbose
    )


def get_model_metadata(
    model_id: str,
    connector_name: str,
    *,
    storage_path: str | None = None
) -> dict:
    """Quick function to get formatted metadata for a model.
    
    >>> import tempfile, os, json
    >>> if 'OPENROUTER_API_KEY' in os.environ:
    ...     from aix.ai_models.manager import get_manager
    ...     with tempfile.NamedTemporaryFile(mode='w+', suffix=".json", delete=False) as temp:
    ...         storage_path = temp.name
    ...         json.dump({'models': []}, temp)
    ...     try:
    ...         manager = get_manager(storage_path=storage_path)
    ...         _ = manager.discover_from_source("openrouter", auto_register=True, verbose=False)
    ...         assert 'openai/gpt-3.5-turbo' in manager.models
    ...         metadata = get_model_metadata("openai/gpt-3.5-turbo", "openrouter", storage_path=storage_path)
    ...         assert 'model' in metadata
    ...     finally:
    ...         os.remove(storage_path)
    """
    manager = get_manager(storage_path=storage_path)
    return manager.get_connector_metadata(model_id, connector_name)
