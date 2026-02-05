"""Model discovery and selection interface for AIX.

Extends the ai_models module with a clean, user-friendly interface for
discovering, filtering, and selecting AI models across providers.

Examples:
    List all models:
    >>> from aix.models import models
    >>> all_models = list(models)  # doctest: +SKIP
    >>> len(all_models) > 0  # doctest: +SKIP
    True

    Get specific model:
    >>> model_info = models['openai/gpt-4o']  # doctest: +SKIP
    >>> model_info.provider  # doctest: +SKIP
    'openai'

    Filter models:
    >>> cheap_models = models.filter(
    ...     custom_filter=lambda m: m.cost_per_token.get('input', 0) < 0.001
    ... )  # doctest: +SKIP

    Use with chat:
    >>> from aix import chat
    >>> chat("Hello", model="gpt-4o-mini")  # doctest: +SKIP
    'Hello! How can I help you?'
"""

from collections.abc import Mapping, Iterator
from typing import Union, Any
from pathlib import Path

# Import from ai_models
from aix.ai_models import (
    Model,
    ModelManager,
    get_manager,
    list_available_models,
    discover_models as _discover_models,
)


class ModelStore(Mapping):
    """User-friendly interface for model discovery and selection.

    Provides a Mapping interface over the ModelManager with convenient
    access patterns and integration with chat/embeddings functions.

    Examples:
        >>> models = ModelStore()  # doctest: +SKIP
        >>> models.discover()  # Fetch available models  # doctest: +SKIP

        >>> # List all models
        >>> list(models)[:5]  # doctest: +SKIP
        ['openai/gpt-4o', 'openai/gpt-4o-mini', ...]

        >>> # Get model info
        >>> info = models['openai/gpt-4o']  # doctest: +SKIP
        >>> info.provider  # doctest: +SKIP
        'openai'

        >>> # Filter models
        >>> openai_models = models.filter(provider='openai')  # doctest: +SKIP
        >>> local_models = models.filter(is_local=True)  # doctest: +SKIP

        >>> # Use with chat
        >>> from aix.chat import chat
        >>> model = models['gpt-4o-mini']  # doctest: +SKIP
        >>> chat("Hello", model=model.id)  # doctest: +SKIP
        'Hello! How can I help you?'
    """

    def __init__(
        self, storage_path: Union[str, Path] = None, auto_discover: bool = False
    ):
        """Initialize model store.

        Args:
            storage_path: Optional path for persistent storage
            auto_discover: If True, automatically discover models on init
        """
        if isinstance(storage_path, str):
            storage_path = Path(storage_path)

        self._manager = get_manager(storage_path=storage_path)
        self._discovered = False

        if auto_discover:
            self.discover()

    def discover(
        self,
        source: str = "openrouter",
        auto_register: bool = True,
        verbose: bool = False,
    ) -> list[Model]:
        """Discover models from a source.

        Args:
            source: Source name ('openrouter', 'ollama', etc.)
            auto_register: If True, add discovered models to registry
            verbose: If True, print progress information

        Returns:
            List of discovered models

        Examples:
            >>> models = ModelStore()  # doctest: +SKIP
            >>> discovered = models.discover('openrouter')  # doctest: +SKIP
            >>> len(discovered) > 100  # doctest: +SKIP
            True
        """
        models_list = self._manager.discover_from_source(
            source, auto_register=auto_register, verbose=verbose
        )
        self._discovered = True
        return models_list

    def __getitem__(self, key: Union[str, dict]) -> Union[Model, list[Model]]:
        """Get model(s) by ID or criteria.

        Args:
            key: Either:
                - Model ID string: 'openai/gpt-4o'
                - Dict of filter criteria: {'provider': 'openai', 'is_local': False}

        Returns:
            Single Model or list of Models

        Examples:
            >>> models = ModelStore()  # doctest: +SKIP
            >>> models.discover(verbose=False)  # doctest: +SKIP
            >>> model = models['openai/gpt-4o']  # doctest: +SKIP
            >>> model.id  # doctest: +SKIP
            'openai/gpt-4o'

            >>> openai_models = models[{'provider': 'openai'}]  # doctest: +SKIP
            >>> len(openai_models) > 0  # doctest: +SKIP
            True
        """
        if isinstance(key, str):
            # Get single model by ID
            return self._manager.models[key]
        elif isinstance(key, dict):
            # Filter by criteria
            return self._manager.list_models(**key)
        else:
            raise TypeError(
                f"Key must be string (model ID) or dict (filter criteria), "
                f"got {type(key)}"
            )

    def __iter__(self) -> Iterator[str]:
        """Iterate over model IDs."""
        return iter(self._manager.models)

    def __len__(self) -> int:
        """Return number of models in store."""
        return len(self._manager.models)

    def __contains__(self, model_id: str) -> bool:
        """Check if model exists."""
        return model_id in self._manager.models

    def filter(
        self,
        *,
        provider: str = None,
        is_local: bool = None,
        min_context_size: int = None,
        max_context_size: int = None,
        has_capabilities: list[str] = None,
        tags: list[str] = None,
        custom_filter: callable = None,
    ) -> list[Model]:
        """Filter models by criteria.

        Args:
            provider: Filter by provider name ('openai', 'anthropic', etc.)
            is_local: Filter by local vs remote
            min_context_size: Minimum context window size
            max_context_size: Maximum context window size
            has_capabilities: Required capabilities
            tags: Required tags
            custom_filter: Custom filter function: f(Model) -> bool

        Returns:
            List of models matching criteria

        Examples:
            >>> models = ModelStore()  # doctest: +SKIP
            >>> models.discover(verbose=False)  # doctest: +SKIP

            >>> # Get OpenAI models
            >>> openai = models.filter(provider='openai')  # doctest: +SKIP

            >>> # Get local models
            >>> local = models.filter(is_local=True)  # doctest: +SKIP

            >>> # Get cheap models
            >>> cheap = models.filter(
            ...     custom_filter=lambda m: m.cost_per_token.get('input', 0) < 0.001
            ... )  # doctest: +SKIP

            >>> # Combine criteria
            >>> good_models = models.filter(
            ...     provider='openai',
            ...     min_context_size=8000,
            ...     custom_filter=lambda m: 'gpt-4' in m.id
            ... )  # doctest: +SKIP
        """
        return self._manager.models.filter(
            provider=provider,
            is_local=is_local,
            min_context_size=min_context_size,
            max_context_size=max_context_size,
            has_capabilities=has_capabilities,
            tags=tags,
            custom_filter=custom_filter,
        )

    def get_info(self, model_id: str) -> Model:
        """Get detailed information about a model.

        Args:
            model_id: Model identifier

        Returns:
            Model object with full metadata

        Examples:
            >>> models = ModelStore()  # doctest: +SKIP
            >>> models.discover(verbose=False)  # doctest: +SKIP
            >>> info = models.get_info('openai/gpt-4o')  # doctest: +SKIP
            >>> info.context_size  # doctest: +SKIP
            128000
        """
        return self._manager.models[model_id]

    def get_connector_metadata(self, model_id: str, connector: str) -> dict[str, Any]:
        """Get connector-specific metadata for a model.

        Args:
            model_id: Model identifier
            connector: Connector name ('openai', 'openrouter', etc.)

        Returns:
            Dict with connector-specific parameters

        Examples:
            >>> models = ModelStore()  # doctest: +SKIP
            >>> meta = models.get_connector_metadata(
            ...     'openai/gpt-4o',
            ...     'openai'
            ... )  # doctest: +SKIP
            >>> meta['model']  # doctest: +SKIP
            'gpt-4o'
        """
        return self._manager.get_connector_metadata(model_id, connector)

    def search(self, query: str) -> list[Model]:
        """Search models by text query.

        Searches in model ID, provider, and tags.

        Args:
            query: Search query

        Returns:
            List of matching models

        Examples:
            >>> models = ModelStore()  # doctest: +SKIP
            >>> models.discover(verbose=False)  # doctest: +SKIP
            >>> results = models.search('gpt-4')  # doctest: +SKIP
            >>> len(results) > 0  # doctest: +SKIP
            True
        """
        query_lower = query.lower()
        return [
            model
            for model in self._manager.models.values()
            if (
                query_lower in model.id.lower()
                or query_lower in model.provider.lower()
                or any(query_lower in tag.lower() for tag in model.tags)
            )
        ]

    def by_provider(self, provider: str) -> list[Model]:
        """Get all models from a specific provider.

        Args:
            provider: Provider name

        Returns:
            List of models

        Examples:
            >>> models = ModelStore()  # doctest: +SKIP
            >>> openai_models = models.by_provider('openai')  # doctest: +SKIP
        """
        return self.filter(provider=provider)

    def by_task(self, task: str) -> list[Model]:
        """Get models suitable for a specific task.

        Args:
            task: Task name ('chat', 'embedding', 'image', etc.)

        Returns:
            List of suitable models

        Examples:
            >>> models = ModelStore()  # doctest: +SKIP
            >>> chat_models = models.by_task('chat')  # doctest: +SKIP
        """
        return self.filter(has_capabilities=[task])

    def recommend(
        self,
        *,
        task: str = "chat",
        max_cost_per_mtok: float = None,
        min_context_size: int = None,
        prefer_local: bool = False,
    ) -> list[Model]:
        """Get recommended models based on requirements.

        Args:
            task: Primary task ('chat', 'embedding', etc.)
            max_cost_per_mtok: Maximum cost per million tokens
            min_context_size: Minimum required context size
            prefer_local: Prefer local models if available

        Returns:
            List of recommended models, sorted by suitability

        Examples:
            >>> models = ModelStore()  # doctest: +SKIP
            >>> recommendations = models.recommend(
            ...     task='chat',
            ...     max_cost_per_mtok=5.0,
            ...     min_context_size=16000
            ... )  # doctest: +SKIP
        """
        candidates = self.filter(
            is_local=True if prefer_local else None,
            min_context_size=min_context_size,
            has_capabilities=[task] if task else None,
        )

        if max_cost_per_mtok is not None:
            candidates = [
                m
                for m in candidates
                if m.cost_per_token.get("input", 0) * 1_000_000 <= max_cost_per_mtok
            ]

        # Sort by cost (cheaper first) if not preferring local
        if not prefer_local:
            candidates.sort(key=lambda m: m.cost_per_token.get("input", float("inf")))

        return candidates

    def __repr__(self) -> str:
        """String representation."""
        return f"ModelStore({len(self)} models)"


# Create singleton instance
models = ModelStore()


def discover_available_models(
    source: str = "openrouter", verbose: bool = True
) -> list[Model]:
    """Discover available models from a source.

    Convenience function that uses the global models instance.

    Args:
        source: Source name ('openrouter', 'ollama', etc.)
        verbose: Print progress information

    Returns:
        List of discovered models

    Examples:
        >>> from aix.models import discover_available_models
        >>> models = discover_available_models('openrouter')  # doctest: +SKIP
        >>> len(models) > 100  # doctest: +SKIP
        True
    """
    return models.discover(source=source, verbose=verbose)


def get_model_info(model_id: str) -> Model:
    """Get information about a specific model.

    Convenience function that uses the global models instance.

    Args:
        model_id: Model identifier

    Returns:
        Model object

    Examples:
        >>> from aix.models import get_model_info
        >>> info = get_model_info('openai/gpt-4o')  # doctest: +SKIP
        >>> info.provider  # doctest: +SKIP
        'openai'
    """
    return models.get_info(model_id)


def find_models(query: str) -> list[Model]:
    """Search for models matching a query.

    Convenience function that uses the global models instance.

    Args:
        query: Search query

    Returns:
        List of matching models

    Examples:
        >>> from aix.models import find_models
        >>> results = find_models('claude')  # doctest: +SKIP
        >>> any('claude' in m.id.lower() for m in results)  # doctest: +SKIP
        True
    """
    return models.search(query)
