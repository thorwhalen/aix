# aix.ai_models.manager

Unified model management interface.

Provides a high-level API for managing AI models across providers.

### Functions

| [`get_manager`](#aix.ai_models.manager.get_manager)([storage_path])   | Get a pre-configured model manager.   |
|--------------------------------------------------------------------------------|---------------------------------------|

### Classes

| [`ModelManager`](#aix.ai_models.manager.ModelManager)(\*[, storage_path])   | Unified interface for AI model management.   |
|-------------------------------------------------------------------------------------|----------------------------------------------|

### *class* aix.ai_models.manager.ModelManager(, storage_path=None)

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

#### *property* connectors *: [ConnectorRegistry](aix.ai_models.base.html.md#aix.ai_models.base.ConnectorRegistry)*

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
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Model`](aix.ai_models.base.html.md#aix.ai_models.base.Model)]]
* **Returns:**
  Dict mapping source name to list of discovered models

#### discover_from_source(source_name, , auto_register=True, verbose=True)

Discover models from a registered source.

* **Parameters:**
  * **source_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name of registered source
  * **auto_register** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, automatically add discovered models to registry
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Model`](aix.ai_models.base.html.md#aix.ai_models.base.Model)]
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
  [`Model`](aix.ai_models.base.html.md#aix.ai_models.base.Model)

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
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Model`](aix.ai_models.base.html.md#aix.ai_models.base.Model)]

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

#### *property* models *: [ModelRegistry](aix.ai_models.base.html.md#aix.ai_models.base.ModelRegistry)*

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

### aix.ai_models.manager.get_manager(storage_path=None)

Get a pre-configured model manager.

* **Return type:**
  [`ModelManager`](#aix.ai_models.manager.ModelManager)

```pycon
>>> manager = get_manager()
>>> len(manager.connectors) > 0
True
```
