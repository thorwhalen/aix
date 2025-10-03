# AI Model Management Module

## Overview

A Python module for unified management of AI models across multiple providers and deployment methods. Provides a clean, extensible interface following the Mapping/MutableMapping pattern for storage-like interactions.

## Package Structure

```
ai_models/
├── __init__.py          # Public API exports
├── base.py              # Core types (Model, ModelRegistry, Connector, ModelSource)
├── sources.py           # Concrete source and connector implementations
├── manager.py           # Unified ModelManager facade
├── examples.py          # Examples
```

## Design Principles

### 1. Mapping/MutableMapping for Storage

**ModelRegistry** implements `MutableMapping[str, Model]` for intuitive key-value access:

```python
# Instead of explicit methods
registry.create_model("gpt-4", model_data)
registry.get_model("gpt-4")
registry.delete_model("gpt-4")
list(registry.list_models())

# Use natural dict-like interface
registry["gpt-4"] = model_data
model = registry["gpt-4"]
del registry["gpt-4"]
list(registry)
```

**Enhanced `__getitem__`** supports multiple access patterns:
- Single key: `registry["gpt-4"]` → Model
- Multiple keys: `registry[["gpt-4", "claude-3"]]` → list[Model]
- Filter callable: `registry[lambda m: m.is_local]` → list[Model]

### 2. Iterable/Iterator for Collections

Filter methods return `list[Model]` but could yield values for lazy evaluation:

```python
# Current implementation
models = registry.filter(provider="openai")  # Returns list

# Could be optimized to:
def filter(...) -> Iterable[Model]:
    for model in self._models.values():
        if _matches(model):
            yield model
```

### 3. Single Source of Truth (SSOT)

- **Model metadata** lives in `Model` dataclass
- **Registry** is the single source for "my models"
- **Sources** discover external models but don't store them
- **Connectors** format existing model metadata, don't duplicate it

### 4. Dependency Injection

Components are injected rather than hard-coded:

```python
# Manager accepts sources and connectors
manager.register_source("custom", CustomSource())
manager.register_connector(CustomConnector())

# Registry accepts storage backend
registry = ModelRegistry(storage_path=custom_path)
```

### 5. Open-Closed Design

- Extend via **new Sources** (discover from new providers)
- Extend via **new Connectors** (support new clients)
- No need to modify core classes

### 6. Dataclasses for Data

`Model` is a dataclass with clear structure:

```python
@dataclass
class Model:
    id: str
    provider: str
    context_size: int | None = None
    is_local: bool = False
    capabilities: dict[str, Any] = field(default_factory=dict)
    cost_per_token: dict[str, float] = field(default_factory=dict)
    tags: set[str] = field(default_factory=set)
    connector_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    custom_metadata: dict[str, Any] = field(default_factory=dict)
```

## Core Components

### Model

Represents an AI model with:
- **Identity**: `id`, `provider`
- **Capabilities**: `context_size`, `capabilities` dict, `cost_per_token`
- **Deployment**: `is_local` flag
- **Organization**: `tags` for grouping, `custom_metadata` for user annotations
- **Connector info**: `connector_metadata` holds provider-specific IDs/config

### ModelRegistry (MutableMapping[str, Model])

Central storage for models with:
- Dict-like access: `registry[model_id]`
- Enhanced `__getitem__`: supports lists and callables
- Filtering: `registry.filter(provider=..., is_local=..., ...)`
- Persistence: Optional JSON storage
- CRUD operations through Mapping protocol

### ModelSource (ABC)

Discovers models from external sources:
- **OpenRouterSource**: Queries OpenRouter API for cross-provider models
- **OllamaSource**: Lists locally installed Ollama models
- **ProviderAPISource**: Generic wrapper for provider APIs (OpenAI, Anthropic, etc.)
- **Custom sources**: Implement `discover_models()` to add new providers

### Connector (ABC)

Formats model metadata for specific clients:
- **OpenAIConnector**: Format for `openai.ChatCompletion.create()`
- **LangChainConnector**: Format for LangChain's ChatOpenAI, ChatOllama, etc.
- **OllamaConnector**: Format for direct Ollama API calls
- **DSPyConnector**: Format for DSPy framework
- **Custom connectors**: Implement `format_metadata()` for new clients

### ModelManager (Facade)

High-level API that unifies all components:
- Manages registry, sources, and connectors
- `discover_from_source()`: Find new models
- `list_models()`: Filter available models
- `get_connector_metadata()`: Format for specific client
- `add_model_tag()`, `set_custom_metadata()`: Annotate models
- `create_model_group()`: Organize models into named groups

## Usage Patterns

### Basic Workflow

```python
from ai_models import get_manager

# Get pre-configured manager
manager = get_manager(storage_path="~/.config/ai_models.json")

# Discover models
manager.discover_from_source("openrouter", auto_register=True)
manager.discover_from_source("ollama", auto_register=True)

# Filter models
local_models = manager.list_models(is_local=True)
cheap_models = manager.list_models(
    custom_filter=lambda m: m.cost_per_token.get("input", 0) < 0.001
)

# Get connector-specific metadata
openai_meta = manager.get_connector_metadata("gpt-4", "openai")
# Use with: openai.ChatCompletion.create(**openai_meta, messages=[...])
```

### Custom Metadata & Tags

```python
# Add custom annotations
manager.set_custom_metadata("gpt-4", "notes", "Best for complex reasoning")
manager.add_model_tag("gpt-4", "production", "expensive")

# Create groups
manager.create_model_group("production-ready", ["gpt-4", "claude-3"])

# Filter by tags
prod_models = manager.list_models(tags=["production"])
group_models = manager.list_models(tags=["group:production-ready"])
```

### Mall Pattern (Advanced)

For organized multi-dimensional access:

```python
from ai_models.mall import ModelMall

mall = ModelMall(manager)

# Browse by provider
mall["by_provider", "openai"]  # All OpenAI models

# Browse by capability
mall["by_capability", "streaming"]  # All streaming models

# Access collections
mall["local"]   # Local models
mall["remote"]  # Remote models
mall["groups"]  # All named groups
```

### Custom Sources

```python
class MyCustomSource(ModelSource):
    def discover_models(self) -> Iterable[Model]:
        # Your discovery logic
        yield Model(...)

manager.register_source("custom", MyCustomSource())
manager.discover_from_source("custom", auto_register=True)
```

### Custom Connectors

```python
class MyCustomConnector(Connector):
    @property
    def name(self) -> str:
        return "my_client"
    
    def format_metadata(self, model: Model) -> dict[str, Any]:
        # Your formatting logic
        return {"model_id": model.id, ...}

manager.register_connector(MyCustomConnector())
metadata = manager.get_connector_metadata("gpt-4", "my_client")
```

## Key Features

### ✅ Model Discovery
- Query multiple sources (OpenRouter, Ollama, provider APIs)
- Auto-register discovered models
- Extensible source system

### ✅ Flexible Filtering
- Filter by provider, deployment type, capabilities, context size
- Custom filter functions
- Tag-based organization
- Model groups

### ✅ Multi-Connector Support
- Format metadata for different clients (OpenAI, LangChain, DSPy, Ollama)
- Extensible connector system
- Store connector-specific metadata per model

### ✅ Custom Metadata
- Annotate models with arbitrary key-value pairs
- Tag models for organization
- Create named groups
- All metadata persists to JSON

### ✅ Persistent Storage
- Optional JSON file storage
- Automatic save on changes
- Cross-session model registry

## Design Decisions

### Why Mapping/MutableMapping?

Storage-like interactions (CRUD on models) fit naturally into the key-value pattern. Users get:
- Intuitive dict-like API
- Python ecosystem compatibility (can use `dict` methods)
- Type-safe with generic parameters

### Why Separate Sources and Connectors?

**Sources** answer: "What models exist?"
**Connectors** answer: "How do I use this model with client X?"

This separation follows Single Responsibility Principle and allows:
- Add new providers without changing connectors
- Add new clients without changing sources
- Mix and match (any model can work with any connector)

### Why Model IDs Might Not Be Universal?

Different systems use different IDs:
- OpenAI: `"gpt-4"`
- Ollama: `"llama2:7b"`
- HuggingFace: `"meta-llama/Llama-2-7b-hf"`

**Solution**: 
- Store canonical ID in `Model.id` (user's choice, could be OpenRouter's ID)
- Store provider-specific IDs in `Model.connector_metadata[provider]["id"]`
- Connectors use the right ID for their client

Example:
```python
model = Model(
    id="gpt-4",  # Canonical
    provider="openai",
    connector_metadata={
        "openai": {"id": "gpt-4"},
        "openrouter": {"id": "openai/gpt-4"},
        "azure": {"id": "gpt-4-deployment-name"}
    }
)

# Connector picks the right one
openai_meta = openai_connector.format_metadata(model)
# Uses connector_metadata["openai"]["id"]
```

### Why JSON for Persistence?

- Human-readable
- Easy to edit manually
- Widely supported
- Good for metadata (not binary data)

Could extend to SQLite for larger registries or richer queries.

## Extension Points

1. **Add a new model source**: Subclass `ModelSource`
2. **Add a new connector**: Subclass `Connector`
3. **Custom filtering**: Pass `custom_filter` callable
4. **Custom storage backend**: Could make `ModelRegistry` accept a storage adapter
5. **Model versioning**: Add version field to `Model`, filter by version
6. **Model aliases**: Add `aliases: list[str]` field, update `__getitem__` to check aliases

## Future Enhancements

- **Query language**: Support dict-based queries like `{"provider": "openai", "context_size": {"$gte": 8000}}`
- **Model benchmarks**: Add benchmark scores as metadata
- **Cost tracking**: Track actual usage costs per model
- **Model comparison**: Built-in method to compare models side-by-side
- **Async sources**: Support async `discover_models()` for faster concurrent discovery
- **Cache management**: Smart caching of discovery results
- **Model health checks**: Ping models to verify availability

## Alignment with Preferences

✅ **Favor Mapping/MutableMapping**: `ModelRegistry` and `ConnectorRegistry`
✅ **Dataclasses**: `Model` is a dataclass
✅ **Minimal docstrings with doctests**: All public APIs have them
✅ **Facades**: `ModelManager` is the main facade
✅ **SSOT**: `Model` and `ModelRegistry` are the single sources of truth
✅ **Dependency Injection**: Sources and connectors are injected
✅ **Open-Closed**: Extend via new Sources/Connectors without modifying core
✅ **Helper functions**: Internal helpers prefixed with `_` (e.g., `_matches`, `_by_provider`)
✅ **Iterable/Iterator ready**: Return types could be made lazy if needed

## Conclusion

This module provides a clean, extensible architecture for managing AI models. The Mapping-based interface makes it intuitive to use, while the Source/Connector abstractions make it easy to extend. The design follows functional patterns where appropriate and uses SOLID principles for OOP components.