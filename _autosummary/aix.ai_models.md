# aix.ai_models

AI Model Management Module.

A unified interface for managing AI models across multiple providers.

Basic usage:

```pycon
>>> from aix.ai_models import get_manager
>>> manager = get_manager()
>>> _ = manager.discover_from_source("openrouter", auto_register=True, verbose=False)
>>> models = manager.list_models(provider="openai")
```

Custom filtering:

```pycon
>>> cheap_models = manager.list_models(
...     custom_filter=lambda m: m.cost_per_token.get("input", 0) < 0.001
... )
```

Get connector-specific metadata:

```pycon
>>> openai_meta = manager.get_connector_metadata("openai/gpt-4", "openai")
>>> # Use with: openai.ChatCompletion.create(**openai_meta, messages=[...])
```

### Functions

| [`get_manager`](#aix.ai_models.get_manager)([storage_path])   | Get a pre-configured model manager.   |
|--------------------------------------------------------------------------------|---------------------------------------|

### Classes

| [`Model`](#aix.ai_models.Model)(id, provider[, context_size, ...])    | Represents an AI model with its metadata.                        |
|----------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| [`ModelRegistry`](#aix.ai_models.ModelRegistry)(\*[, storage_path])           | Registry for managing AI models using Mapping interface.         |
| [`ModelSource`](#aix.ai_models.ModelSource)()                               | Abstract base for model discovery sources.                       |
| [`Connector`](#aix.ai_models.Connector)()                                 | Abstract base for model connectors/clients.                      |
| [`ConnectorRegistry`](#aix.ai_models.ConnectorRegistry)()                         | Registry for managing model connectors.                          |
| [`OpenRouterSource`](#aix.ai_models.OpenRouterSource)(\*[, timeout])             | Discover models from OpenRouter's public API.                    |
| [`OllamaSource`](#aix.ai_models.OllamaSource)(\*[, base_url, timeout])       | Discover locally installed Ollama models.                        |
| [`ProviderAPISource`](#aix.ai_models.ProviderAPISource)(provider_name, \*[, ...]) | Discover models from a provider's API (OpenAI, Anthropic, etc.). |
| [`OpenAIConnector`](#aix.ai_models.OpenAIConnector)()                           | Format metadata for OpenAI Python client.                        |
| [`OpenRouterConnector`](#aix.ai_models.OpenRouterConnector)()                       | Connector for OpenRouter API.                                    |
| [`LangChainConnector`](#aix.ai_models.LangChainConnector)()                        | Connector for LangChain library.                                 |
| [`OllamaConnector`](#aix.ai_models.OllamaConnector)()                           | Format metadata for direct Ollama API calls.                     |
| [`DSPyConnector`](#aix.ai_models.DSPyConnector)()                             | Format metadata for DSPy framework.                              |
| [`ModelManager`](#aix.ai_models.ModelManager)(\*[, storage_path])            | Unified interface for AI model management.                       |

### *class* aix.ai_models.Connector

Bases: [`ABC`](https://docs.python.org/3/library/abc.html#abc.ABC)

Abstract base for model connectors/clients.

#### *abstractmethod* format_metadata(model)

Format model metadata for this connector.

Returns a dict that can be used to instantiate/connect via this connector.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

#### *abstract property* name *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

Unique identifier for this connector.

### *class* aix.ai_models.ConnectorRegistry

Bases: [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Connector`](aix.ai_models.base.md#aix.ai_models.base.Connector)]

Registry for managing model connectors.

```pycon
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
```

### *class* aix.ai_models.DSPyConnector

Bases: [`Connector`](aix.ai_models.base.md#aix.ai_models.base.Connector)

Format metadata for DSPy framework.

#### format_metadata(model)

Format for DSPy.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

```pycon
>>> connector = DSPyConnector()
>>> model = Model(id="gpt-4", provider="openai")
>>> metadata = connector.format_metadata(model)
>>> "model" in metadata
True
```

#### *property* name *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

Unique identifier for this connector.

### *class* aix.ai_models.LangChainConnector

Bases: [`Connector`](aix.ai_models.base.md#aix.ai_models.base.Connector)

Connector for LangChain library.

#### format_metadata(model)

Format for LangChain.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

```pycon
>>> connector = LangChainConnector()
>>> model = Model(id="gpt-4", provider="openai")
>>> metadata = connector.format_metadata(model)
>>> metadata["model_name"]
'gpt-4'
```

#### *property* name *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

Unique identifier for this connector.

### *class* aix.ai_models.Model(id, provider, context_size=None, is_local=False, capabilities=<factory>, cost_per_token=<factory>, tags=<factory>, connector_metadata=<factory>, custom_metadata=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Represents an AI model with its metadata.

```pycon
>>> model = Model(
...     id="gpt-4",
...     provider="openai",
...     context_size=8192,
...     is_local=False
... )
>>> model.id
'gpt-4'
```

#### matches_filter(\*\*criteria)

Check if model matches given criteria.

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

```pycon
>>> model = Model(id="gpt-4", provider="openai", is_local=False)
>>> model.matches_filter(provider="openai")
True
>>> model.matches_filter(is_local=True)
False
```

#### to_dict()

Convert model to dictionary representation.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### *class* aix.ai_models.ModelManager(, storage_path=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Unified interface for AI model management.

Combines model registry, sources, and connectors into a single facade.

```pycon
>>> manager = ModelManager()
>>> manager.register_connector(OpenAIConnector())
>>> len(manager.connectors)
1
```

#### add_model_tag(model_id, \*tags)

Add tags to a model.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

```pycon
>>> manager = ModelManager()
>>> manager.models["gpt-4"] = Model(id="gpt-4", provider="openai")
>>> manager.add_model_tag("gpt-4", "fast", "expensive")
>>> "fast" in manager.models["gpt-4"].tags
True
```

#### *property* connectors *: [ConnectorRegistry](aix.ai_models.base.md#aix.ai_models.base.ConnectorRegistry)*

Access to connector registry.

#### create_model_group(group_name, model_ids)

Create a named group of models using tags.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

```pycon
>>> manager = ModelManager()
>>> manager.models["gpt-4"] = Model(id="gpt-4", provider="openai")
>>> manager.models["claude-3"] = Model(id="claude-3", provider="anthropic")
>>> manager.create_model_group("premium", ["gpt-4", "claude-3"])
>>> premium = manager.list_models(tags=["group:premium"])
>>> len(premium)
2
```

#### discover_all(, auto_register=True)

Discover models from all registered sources.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Model`](aix.ai_models.base.md#aix.ai_models.base.Model)]]
* **Returns:**
  Dict mapping source name to list of discovered models

#### discover_from_source(source_name, , auto_register=True, verbose=True)

Discover models from a registered source.

* **Parameters:**
  * **source_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of registered source
  * **auto_register** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, automatically add discovered models to registry
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Model`](aix.ai_models.base.md#aix.ai_models.base.Model)]
* **Returns:**
  List of discovered models

#### get_connector_metadata(model_id, connector_name)

Get formatted metadata for a specific connector.

* **Parameters:**
  * **model_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – ID of the model
  * **connector_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of the connector to format for
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]
* **Returns:**
  Dict with metadata formatted for the specified connector

```pycon
>>> manager = ModelManager()
>>> manager.models["gpt-4"] = Model(id="gpt-4", provider="openai")
>>> manager.register_connector(OpenAIConnector())
>>> meta = manager.get_connector_metadata("gpt-4", "openai")
>>> meta["model"]
'gpt-4'
```

#### get_model(model_id)

Get a specific model by ID.

* **Return type:**
  [`Model`](aix.ai_models.base.md#aix.ai_models.base.Model)

```pycon
>>> manager = ModelManager()
>>> manager.models["gpt-4"] = Model(id="gpt-4", provider="openai")
>>> model = manager.get_model("gpt-4")
>>> model.id
'gpt-4'
```

#### list_model_groups()

List all model groups.

* **Returns:**
  ‘ prefix)
* **Return type:**
  [`set`](https://docs.python.org/3/builtins/stdtypes.html#set)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### list_models(, provider=None, is_local=None, min_context_size=None, max_context_size=None, has_capabilities=None, tags=None, custom_filter=None)

List models with optional filtering.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Model`](aix.ai_models.base.md#aix.ai_models.base.Model)]

```pycon
>>> manager = ModelManager()
>>> manager.models["gpt-4"] = Model(id="gpt-4", provider="openai")
>>> manager.models["llama2"] = Model(id="llama2", provider="ollama", is_local=True)
>>> local = manager.list_models(is_local=True)
>>> len(local)
1
>>> local[0].id
'llama2'
```

#### list_sources()

List registered source names.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### *property* models *: [ModelRegistry](aix.ai_models.base.md#aix.ai_models.base.ModelRegistry)*

Access to model registry for direct manipulation.

```pycon
>>> manager = ModelManager()
>>> manager.models["test"] = Model(id="test", provider="test")
>>> "test" in manager.models
True
```

#### register_connector(connector)

Register a model connector.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

```pycon
>>> manager = ModelManager()
>>> connector = OpenAIConnector()
>>> manager.register_connector(connector)
>>> connector.name in manager.connectors
True
```

#### register_source(name, source)

Register a model discovery source.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

```pycon
>>> manager = ModelManager()
>>> source = OpenRouterSource()
>>> manager.register_source("openrouter", source)
>>> "openrouter" in manager.list_sources()
True
```

#### set_custom_metadata(model_id, key, value)

Set custom metadata on a model.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

```pycon
>>> manager = ModelManager()
>>> manager.models["gpt-4"] = Model(id="gpt-4", provider="openai")
>>> manager.set_custom_metadata("gpt-4", "my_note", "Use for production")
>>> manager.models["gpt-4"].custom_metadata["my_note"]
'Use for production'
```

### *class* aix.ai_models.ModelRegistry(, storage_path=None)

Bases: [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Model`](aix.ai_models.base.md#aix.ai_models.base.Model)]

Registry for managing AI models using Mapping interface.

```pycon
>>> registry = ModelRegistry()
>>> registry["gpt-4"] = Model(id="gpt-4", provider="openai")
>>> "gpt-4" in registry
True
>>> len(registry)
1
```

#### filter(, provider=None, is_local=None, min_context_size=None, max_context_size=None, has_capabilities=None, tags=None, custom_filter=None)

Filter models by multiple criteria.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Model`](aix.ai_models.base.md#aix.ai_models.base.Model)]

```pycon
>>> registry = ModelRegistry()
>>> registry["gpt-4"] = Model(id="gpt-4", provider="openai", context_size=8192)
>>> registry["llama2"] = Model(id="llama2", provider="ollama", is_local=True)
>>> local_models = registry.filter(is_local=True)
>>> len(local_models)
1
```

### *class* aix.ai_models.ModelSource

Bases: [`ABC`](https://docs.python.org/3/library/abc.html#abc.ABC)

Abstract base for model discovery sources.

#### *abstractmethod* discover_models()

Discover available models from this source.

Yields Model instances for each discovered model.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Model`](aix.ai_models.base.md#aix.ai_models.base.Model)]

### *class* aix.ai_models.OllamaConnector

Bases: [`Connector`](aix.ai_models.base.md#aix.ai_models.base.Connector)

Format metadata for direct Ollama API calls.

#### format_metadata(model)

Format for Ollama API.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

```pycon
>>> connector = OllamaConnector()
>>> model = Model(id="ollama/llama2", provider="ollama", is_local=True)
>>> model.connector_metadata["ollama"] = {"name": "llama2"}
>>> metadata = connector.format_metadata(model)
>>> metadata["model"]
'llama2'
```

#### *property* name *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

Unique identifier for this connector.

### *class* aix.ai_models.OllamaSource(, base_url='http://localhost:11434', timeout=10)

Bases: [`ModelSource`](aix.ai_models.base.md#aix.ai_models.base.ModelSource)

Discover locally installed Ollama models.

Queries the Ollama API for models available on the local machine.

#### discover_models()

Fetch locally installed Ollama models.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Model`](aix.ai_models.base.md#aix.ai_models.base.Model)]

```pycon
>>> source = OllamaSource()
>>> models = list(source.discover_models())  # May be empty if Ollama not running
>>> all(m.is_local for m in models)
True
```

### *class* aix.ai_models.OpenAIConnector

Bases: [`Connector`](aix.ai_models.base.md#aix.ai_models.base.Connector)

Format metadata for OpenAI Python client.

Provides model metadata in format expected by openai.ChatCompletion.create().

#### format_metadata(model)

Format model metadata for this connector.

Returns a dict that can be used to instantiate/connect via this connector.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

#### *property* name *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

Unique identifier for this connector.

### *class* aix.ai_models.OpenRouterConnector

Bases: [`Connector`](aix.ai_models.base.md#aix.ai_models.base.Connector)

Connector for OpenRouter API.

#### format_metadata(model)

Formats metadata for OpenRouter, which uses the model ID directly.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

#### *property* name *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

Unique identifier for this connector.

### *class* aix.ai_models.OpenRouterSource(, timeout=30)

Bases: [`ModelSource`](aix.ai_models.base.md#aix.ai_models.base.ModelSource)

Discover models from OpenRouter’s public API.

OpenRouter maintains a comprehensive registry of models across providers.

#### discover_models()

Fetch models from OpenRouter API.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Model`](aix.ai_models.base.md#aix.ai_models.base.Model)]

```pycon
>>> source = OpenRouterSource()
>>> models = list(source.discover_models())
>>> len(models) > 0  # Should find many models
True
```

### *class* aix.ai_models.ProviderAPISource(provider_name, , api_key=None, base_url='https://api.openai.com/v1', timeout=30)

Bases: [`ModelSource`](aix.ai_models.base.md#aix.ai_models.base.ModelSource)

Discover models from a provider’s API (OpenAI, Anthropic, etc.).

Generic source for providers with /v1/models endpoint.

#### discover_models()

Fetch models from provider API.

Requires valid API key for the provider.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Model`](aix.ai_models.base.md#aix.ai_models.base.Model)]

### aix.ai_models.get_manager(storage_path=None)

Get a pre-configured model manager.

* **Return type:**
  [`ModelManager`](aix.ai_models.manager.md#aix.ai_models.manager.ModelManager)

```pycon
>>> manager = get_manager()
>>> len(manager.connectors) > 0
True
```

### Modules

| [`base`](aix.ai_models.base.md#module-aix.ai_models.base)       | Core types for AI model management.                       |
|---------------------------------------------------------------------------------------|-----------------------------------------------------------|
| [`manager`](aix.ai_models.manager.md#module-aix.ai_models.manager) | Unified model management interface.                       |
| [`sources`](aix.ai_models.sources.md#module-aix.ai_models.sources) | Concrete implementations of model sources and connectors. |
