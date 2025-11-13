"""Core types for AI model management.

This module provides a unified interface for managing, discovering, and
connecting to AI models across multiple providers and deployment methods.
"""

from dataclasses import dataclass, field, asdict
from typing import Any
from collections.abc import Mapping, MutableMapping, Iterator, Callable, Iterable
from abc import ABC, abstractmethod
import json
from pathlib import Path


@dataclass
class Model:
    """Represents an AI model with its metadata.
    
    >>> model = Model(
    ...     id="gpt-4",
    ...     provider="openai",
    ...     context_size=8192,
    ...     is_local=False
    ... )
    >>> model.id
    'gpt-4'
    """
    id: str
    provider: str
    context_size: int | None = None
    is_local: bool = False
    capabilities: dict[str, Any] = field(default_factory=dict)
    cost_per_token: dict[str, float] = field(default_factory=dict)
    tags: set[str] = field(default_factory=set)
    connector_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    custom_metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary representation."""
        return asdict(self)
    
    def matches_filter(self, **criteria) -> bool:
        """Check if model matches given criteria.
        
        >>> model = Model(id="gpt-4", provider="openai", is_local=False)
        >>> model.matches_filter(provider="openai")
        True
        >>> model.matches_filter(is_local=True)
        False
        """
        for key, value in criteria.items():
            if not hasattr(self, key):
                return False
            if getattr(self, key) != value:
                return False
        return True


class ModelRegistry(MutableMapping[str, Model]):
    """Registry for managing AI models using Mapping interface.
    
    >>> registry = ModelRegistry()
    >>> registry["gpt-4"] = Model(id="gpt-4", provider="openai")
    >>> "gpt-4" in registry
    True
    >>> len(registry)
    1
    """
    
    def __init__(self, *, storage_path: Path | None = None):
        """Initialize registry with optional persistent storage."""
        self._models: dict[str, Model] = {}
        self._storage_path = storage_path
        if storage_path and storage_path.exists():
            self._load()
    
    def __setitem__(self, model_id: str, model: Model) -> None:
        """Add or update a model in the registry."""
        self._models[model_id] = model
        if self._storage_path:
            self._save()
    
    def __getitem__(self, key: str | list[str] | Callable[[Model], bool]) -> Model | list[Model]:
        """Get model(s) by ID, list of IDs, or filter function.
        
        Supports:
        - Single ID: registry["gpt-4"]
        - Multiple IDs: registry[["gpt-4", "claude-3"]]
        - Filter function: registry[lambda m: m.is_local]
        """
        if isinstance(key, str):
            return self._models[key]
        elif isinstance(key, list):
            return [self._models[k] for k in key if k in self._models]
        elif callable(key):
            return [m for m in self._models.values() if key(m)]
        else:
            raise TypeError(f"Unsupported key type: {type(key)}")
    
    def __delitem__(self, model_id: str) -> None:
        """Remove a model from the registry."""
        del self._models[model_id]
        if self._storage_path:
            self._save()
    
    def __iter__(self) -> Iterator[str]:
        """Iterate over model IDs."""
        yield from self._models.keys()
    
    def __len__(self) -> int:
        """Return number of models in registry."""
        return len(self._models)
    
    def filter(
        self,
        *,
        provider: str | None = None,
        is_local: bool | None = None,
        min_context_size: int | None = None,
        max_context_size: int | None = None,
        has_capabilities: Iterable[str] | None = None,
        tags: Iterable[str] | None = None,
        custom_filter: Callable[[Model], bool] | None = None
    ) -> list[Model]:
        """Filter models by multiple criteria.
        
        >>> registry = ModelRegistry()
        >>> registry["gpt-4"] = Model(id="gpt-4", provider="openai", context_size=8192)
        >>> registry["llama2"] = Model(id="llama2", provider="ollama", is_local=True)
        >>> local_models = registry.filter(is_local=True)
        >>> len(local_models)
        1
        """
        def _matches(model: Model) -> bool:
            if provider and model.provider != provider:
                return False
            if is_local is not None and model.is_local != is_local:
                return False
            if min_context_size and (not model.context_size or model.context_size < min_context_size):
                return False
            if max_context_size and (not model.context_size or model.context_size > max_context_size):
                return False
            if has_capabilities:
                for cap in has_capabilities:
                    if not model.capabilities.get(cap):
                        return False
            if tags and not model.tags.issuperset(tags):
                return False
            if custom_filter and not custom_filter(model):
                return False
            return True
        
        return [m for m in self._models.values() if _matches(m)]
    
    def _load(self) -> None:
        """Load models from persistent storage."""
        if not self._storage_path:
            return
        
        with open(self._storage_path) as f:
            data = json.load(f)
            for model_data in data['models']:
                # Reconstruct set for tags
                model_data['tags'] = set(model_data.get('tags', []))
                model = Model(**model_data)
                self._models[model.id] = model
    
    def _save(self) -> None:
        """Save models to persistent storage."""
        if not self._storage_path:
            return
        
        models_data = []
        for model in self._models.values():
            model_dict = model.to_dict()
            # Convert set to list for JSON serialization
            model_dict['tags'] = list(model_dict['tags'])
            models_data.append(model_dict)
        
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._storage_path, 'w') as f:
            json.dump({'models': models_data}, f, indent=2)


class ModelSource(ABC):
    """Abstract base for model discovery sources."""
    
    @abstractmethod
    def discover_models(self) -> Iterable[Model]:
        """Discover available models from this source.
        
        Yields Model instances for each discovered model.
        """
        pass


class Connector(ABC):
    """Abstract base for model connectors/clients."""
    
    @abstractmethod
    def format_metadata(self, model: Model) -> dict[str, Any]:
        """Format model metadata for this connector.
        
        Returns a dict that can be used to instantiate/connect via this connector.
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this connector."""
        pass


class ConnectorRegistry(MutableMapping[str, Connector]):
    """Registry for managing model connectors.
    
    >>> registry = ConnectorRegistry()
    >>> class MyConnector(Connector):
    ...     @property
    ...     def name(self) -> str:
    ...         return "my_connector"
    ...     def format_metadata(self, model: Model) -> dict[str, Any]:
    ...         return {"model": model.id}
    >>> connector = MyConnector()
    >>> registry[connector.name] = connector
    >>> "my_connector" in registry
    True
    """
    
    def __init__(self):
        self._connectors: dict[str, Connector] = {}
    
    def __setitem__(self, name: str, connector: Connector) -> None:
        self._connectors[name] = connector
    
    def __getitem__(self, name: str) -> Connector:
        return self._connectors[name]
    
    def __delitem__(self, name: str) -> None:
        del self._connectors[name]
    
    def __iter__(self) -> Iterator[str]:
        yield from self._connectors.keys()
    
    def __len__(self) -> int:
        return len(self._connectors)
