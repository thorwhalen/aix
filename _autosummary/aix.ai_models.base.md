# aix.ai_models.base

Core types for AI model management.

This module provides a unified interface for managing, discovering, and
connecting to AI models across multiple providers and deployment methods.

### Classes

| [`Connector`](#aix.ai_models.base.Connector)()                              | Abstract base for model connectors/clients.              |
|-------------------------------------------------------------------------------------------|----------------------------------------------------------|
| [`ConnectorRegistry`](#aix.ai_models.base.ConnectorRegistry)()                      | Registry for managing model connectors.                  |
| [`Model`](#aix.ai_models.base.Model)(id, provider[, context_size, ...]) | Represents an AI model with its metadata.                |
| [`ModelRegistry`](#aix.ai_models.base.ModelRegistry)(\*[, storage_path])        | Registry for managing AI models using Mapping interface. |
| [`ModelSource`](#aix.ai_models.base.ModelSource)()                            | Abstract base for model discovery sources.               |

### *class* aix.ai_models.base.Connector

Bases: [`ABC`](https://docs.python.org/3/library/abc.html#abc.ABC)

Abstract base for model connectors/clients.

#### *abstractmethod* format_metadata(model)

Format model metadata for this connector.

Returns a dict that can be used to instantiate/connect via this connector.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

#### *abstract property* name *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

Unique identifier for this connector.

### *class* aix.ai_models.base.ConnectorRegistry

Bases: [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Connector`](#aix.ai_models.base.Connector)]

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

### *class* aix.ai_models.base.Model(id, provider, context_size=None, is_local=False, capabilities=<factory>, cost_per_token=<factory>, tags=<factory>, connector_metadata=<factory>, custom_metadata=<factory>)

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

### *class* aix.ai_models.base.ModelRegistry(, storage_path=None)

Bases: [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Model`](#aix.ai_models.base.Model)]

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
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Model`](#aix.ai_models.base.Model)]

```pycon
>>> registry = ModelRegistry()
>>> registry["gpt-4"] = Model(id="gpt-4", provider="openai", context_size=8192)
>>> registry["llama2"] = Model(id="llama2", provider="ollama", is_local=True)
>>> local_models = registry.filter(is_local=True)
>>> len(local_models)
1
```

### *class* aix.ai_models.base.ModelSource

Bases: [`ABC`](https://docs.python.org/3/library/abc.html#abc.ABC)

Abstract base for model discovery sources.

#### *abstractmethod* discover_models()

Discover available models from this source.

Yields Model instances for each discovered model.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Model`](#aix.ai_models.base.Model)]
