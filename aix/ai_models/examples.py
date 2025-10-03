"""Usage examples for the model management module.

This module demonstrates common patterns and use cases.
"""

from pathlib import Path
from typing import Any, Iterable, Callable
from aix.ai_models.base import ModelSource, Model
from aix.ai_models.manager import ModelManager

# === Basic Usage ===


def example_basic_usage():
    """Basic model management workflow."""
    # Create manager with persistent storage
    manager = get_manager(storage_path="~/.config/ai_models.json")

    # Discover models from OpenRouter
    print("Discovering models from OpenRouter...")
    openrouter_models = manager.discover_from_source("openrouter", auto_register=True)
    print(f"Found {len(openrouter_models)} models from OpenRouter")

    # Discover local Ollama models
    print("\nDiscovering local Ollama models...")
    ollama_models = manager.discover_from_source("ollama", auto_register=True)
    print(f"Found {len(ollama_models)} local models")

    # List all available models
    print(f"\nTotal models in registry: {len(manager.models)}")

    # Filter for specific models
    local_models = manager.list_models(is_local=True)
    print(f"Local models: {[m.id for m in local_models]}")

    openai_models = manager.list_models(provider="openai")
    print(f"OpenAI models: {[m.id for m in openai_models[:5]]}")

    # Get metadata for a specific model and connector
    if "gpt-4" in manager.models:
        openai_meta = manager.get_connector_metadata("gpt-4", "openai")
        print(f"\nOpenAI metadata for gpt-4: {openai_meta}")

        langchain_meta = manager.get_connector_metadata("gpt-4", "langchain")
        print(f"LangChain metadata for gpt-4: {langchain_meta}")


def example_custom_metadata():
    """Working with custom metadata and tags."""
    manager = get_manager()

    # Add a custom model manually
    from base import Model

    manager.models["my-fine-tuned-gpt"] = Model(
        id="my-fine-tuned-gpt", provider="openai", context_size=8192, is_local=False
    )

    # Add custom metadata
    manager.set_custom_metadata(
        "my-fine-tuned-gpt", "training_data", "Customer support conversations"
    )
    manager.set_custom_metadata(
        "my-fine-tuned-gpt",
        "performance_notes",
        "Better at formal tone than base model",
    )

    # Add tags
    manager.add_model_tag("my-fine-tuned-gpt", "production", "customer-facing")

    # Create model groups
    manager.create_model_group("production-ready", ["my-fine-tuned-gpt", "gpt-4"])

    # Filter by tags
    prod_models = manager.list_models(tags=["production"])
    print(f"Production models: {[m.id for m in prod_models]}")

    # Filter by group
    group_models = manager.list_models(tags=["group:production-ready"])
    print(f"Production-ready group: {[m.id for m in group_models]}")


def example_advanced_filtering():
    """Advanced filtering with custom functions."""
    manager = get_manager()

    # Discover models first
    manager.discover_from_source("openrouter", auto_register=True)

    # Filter with multiple criteria
    affordable_models = manager.list_models(
        max_context_size=16000,
        custom_filter=lambda m: (
            m.cost_per_token.get("input", 0) < 0.001
            and m.capabilities.get("streaming", False)
        ),
    )
    print(f"Affordable streaming models: {len(affordable_models)}")

    # Complex filtering with callable
    def _is_suitable_for_task(model: Model) -> bool:
        """Check if model is suitable for a specific task."""
        has_function_calling = model.capabilities.get("function_calling", False)
        has_sufficient_context = (model.context_size or 0) >= 32000
        is_affordable = model.cost_per_token.get("input", float('inf')) < 0.01
        return has_function_calling and has_sufficient_context and is_affordable

    suitable_models = manager.list_models(custom_filter=_is_suitable_for_task)
    print(f"Models suitable for task: {[m.id for m in suitable_models]}")


def example_connector_usage():
    """Using connectors to format metadata for different clients."""
    manager = get_manager()

    # Assume we have a model registered
    model_id = "gpt-4"

    # Get metadata for OpenAI Python client
    openai_meta = manager.get_connector_metadata(model_id, "openai")
    # Use with: openai.ChatCompletion.create(**openai_meta, messages=[...])

    # Get metadata for LangChain
    langchain_meta = manager.get_connector_metadata(model_id, "langchain")
    # Use with: ChatOpenAI(**{k: v for k, v in langchain_meta.items() if k != '_class'})

    # Get metadata for DSPy
    dspy_meta = manager.get_connector_metadata(model_id, "dspy")
    # Use with: dspy.OpenAI(**{k: v for k, v in dspy_meta.items() if k != '_class'})

    print(f"OpenAI format: {openai_meta}")
    print(f"LangChain format: {langchain_meta}")
    print(f"DSPy format: {dspy_meta}")


# === Advanced Pattern: Model Mall ===


class ModelMall:
    """A 'mall' of model collections organized by category.

    Extends the Mapping pattern to group models by different dimensions.

    >>> from aix.ai_models.manager import get_manager
    >>> manager = get_manager()
    >>> mall = ModelMall(manager)
    >>> list(mall.keys())  # Shows available collections
    ['by_provider', 'by_capability', 'local', 'remote', 'groups']
    """

    def __init__(self, manager: ModelManager):
        self._manager = manager

    def __getitem__(self, key: str | tuple[str, ...]) -> Any:
        """Access model collections by key or nested keys.

        Supports:
        - mall['by_provider'] -> dict of providers
        - mall['by_provider', 'openai'] -> list of OpenAI models
        - mall['local'] -> list of local models
        """
        if isinstance(key, tuple):
            # Nested access
            return self._nested_getitem(key)

        # Single key access
        if key == "by_provider":
            return self._by_provider()
        elif key == "by_capability":
            return self._by_capability()
        elif key == "local":
            return self._manager.list_models(is_local=True)
        elif key == "remote":
            return self._manager.list_models(is_local=False)
        elif key == "groups":
            return {
                name: self._manager.list_models(tags=[f"group:{name}"])
                for name in self._manager.list_model_groups()
            }
        else:
            raise KeyError(f"Unknown collection: {key}")

    def _nested_getitem(self, keys: tuple[str, ...]) -> Any:
        """Handle nested key access."""
        if len(keys) == 2:
            collection, item = keys
            if collection == "by_provider":
                return self._manager.list_models(provider=item)
            elif collection == "by_capability":
                return self._manager.list_models(has_capabilities=[item])
            elif collection == "groups":
                return self._manager.list_models(tags=[f"group:{item}"])
        raise KeyError(f"Invalid nested key: {keys}")

    def _by_provider(self) -> dict[str, list[Model]]:
        """Group models by provider."""
        providers = {}
        for model in self._manager.models.values():
            if model.provider not in providers:
                providers[model.provider] = []
            providers[model.provider].append(model)
        return providers

    def _by_capability(self) -> dict[str, list[Model]]:
        """Group models by capability."""
        capabilities = {}
        for model in self._manager.models.values():
            for cap in model.capabilities:
                if cap not in capabilities:
                    capabilities[cap] = []
                capabilities[cap].append(model)
        return capabilities

    def keys(self):
        """List available collections."""
        return ["by_provider", "by_capability", "local", "remote", "groups"]


def example_mall_usage():
    """Using the ModelMall pattern for organized access."""
    manager = get_manager()
    manager.discover_from_source("openrouter", auto_register=True)
    manager.discover_from_source("ollama", auto_register=True)

    mall = ModelMall(manager)

    # Browse available collections
    print(f"Available collections: {list(mall.keys())}")

    # Access models by provider
    providers = mall["by_provider"]
    print(f"Providers: {list(providers.keys())}")

    # Direct nested access
    openai_models = mall["by_provider", "openai"]
    print(f"OpenAI models: {len(openai_models)}")

    # Access by capability
    streaming_models = mall["by_capability", "streaming"]
    print(f"Models with streaming: {len(streaming_models)}")

    # Access local vs remote
    local_models = mall["local"]
    remote_models = mall["remote"]
    print(f"Local: {len(local_models)}, Remote: {len(remote_models)}")


# === Custom Source Example ===


class HuggingFaceSource(ModelSource):
    """Example custom source for HuggingFace Hub models.

    Demonstrates how to add a new model source.
    """

    def __init__(self, *, task: str = "text-generation", limit: int = 10):
        self._task = task
        self._limit = limit

    def discover_models(self) -> Iterable[Model]:
        """Discover models from HuggingFace Hub."""
        try:
            from huggingface_hub import HfApi

            api = HfApi()

            models = api.list_models(filter=self._task, limit=self._limit)

            for hf_model in models:
                yield Model(
                    id=f"hf/{hf_model.modelId}",
                    provider="huggingface",
                    is_local=False,
                    connector_metadata={
                        "huggingface": {
                            "model_id": hf_model.modelId,
                            "task": self._task,
                        }
                    },
                    custom_metadata={
                        "downloads": getattr(hf_model, "downloads", 0),
                        "tags": getattr(hf_model, "tags", []),
                    },
                )
        except ImportError:
            print("huggingface_hub not installed")
            return


def example_custom_source():
    """Using a custom source."""
    manager = get_manager()

    # Register custom source
    hf_source = HuggingFaceSource(task="text-generation", limit=5)
    manager.register_source("huggingface", hf_source)

    # Discover from custom source
    hf_models = manager.discover_from_source("huggingface", auto_register=True)
    print(f"Found {len(hf_models)} HuggingFace models")

    # List all HuggingFace models
    all_hf = manager.list_models(provider="huggingface")
    for model in all_hf:
        print(f"  {model.id} - {model.custom_metadata.get('downloads', 0)} downloads")


if __name__ == "__main__":
    # Run examples
    print("=== Basic Usage ===")
    example_basic_usage()

    print("\n=== Custom Metadata ===")
    example_custom_metadata()

    print("\n=== Advanced Filtering ===")
    example_advanced_filtering()

    print("\n=== Mall Pattern ===")
    example_mall_usage()

    print("\n=== Custom Source ===")
    example_custom_source()
