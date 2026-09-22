# aix.ai_models.sources

Concrete implementations of model sources and connectors.

This module provides ready-to-use sources for discovering models
and connectors for formatting metadata for different clients.

### Classes

| [`DSPyConnector`](#aix.ai_models.sources.DSPyConnector)()                             | Format metadata for DSPy framework.                              |
|----------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| [`LangChainConnector`](#aix.ai_models.sources.LangChainConnector)()                        | Connector for LangChain library.                                 |
| [`OllamaConnector`](#aix.ai_models.sources.OllamaConnector)()                           | Format metadata for direct Ollama API calls.                     |
| [`OllamaSource`](#aix.ai_models.sources.OllamaSource)(\*[, base_url, timeout])       | Discover locally installed Ollama models.                        |
| [`OpenAIConnector`](#aix.ai_models.sources.OpenAIConnector)()                           | Format metadata for OpenAI Python client.                        |
| [`OpenRouterConnector`](#aix.ai_models.sources.OpenRouterConnector)()                       | Connector for OpenRouter API.                                    |
| [`OpenRouterSource`](#aix.ai_models.sources.OpenRouterSource)(\*[, timeout])             | Discover models from OpenRouter's public API.                    |
| [`ProviderAPISource`](#aix.ai_models.sources.ProviderAPISource)(provider_name, \*[, ...]) | Discover models from a provider's API (OpenAI, Anthropic, etc.). |

### *class* aix.ai_models.sources.DSPyConnector

Bases: [`Connector`](aix.ai_models.base.html.md#aix.ai_models.base.Connector)

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

### *class* aix.ai_models.sources.LangChainConnector

Bases: [`Connector`](aix.ai_models.base.html.md#aix.ai_models.base.Connector)

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

### *class* aix.ai_models.sources.OllamaConnector

Bases: [`Connector`](aix.ai_models.base.html.md#aix.ai_models.base.Connector)

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

### *class* aix.ai_models.sources.OllamaSource(, base_url='http://localhost:11434', timeout=10)

Bases: [`ModelSource`](aix.ai_models.base.html.md#aix.ai_models.base.ModelSource)

Discover locally installed Ollama models.

Queries the Ollama API for models available on the local machine.

#### discover_models()

Fetch locally installed Ollama models.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Model`](aix.ai_models.base.html.md#aix.ai_models.base.Model)]

```pycon
>>> source = OllamaSource()
>>> models = list(source.discover_models())  # May be empty if Ollama not running
>>> all(m.is_local for m in models)
True
```

### *class* aix.ai_models.sources.OpenAIConnector

Bases: [`Connector`](aix.ai_models.base.html.md#aix.ai_models.base.Connector)

Format metadata for OpenAI Python client.

Provides model metadata in format expected by openai.ChatCompletion.create().

#### format_metadata(model)

Format model metadata for this connector.

Returns a dict that can be used to instantiate/connect via this connector.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

#### *property* name *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

Unique identifier for this connector.

### *class* aix.ai_models.sources.OpenRouterConnector

Bases: [`Connector`](aix.ai_models.base.html.md#aix.ai_models.base.Connector)

Connector for OpenRouter API.

#### format_metadata(model)

Formats metadata for OpenRouter, which uses the model ID directly.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

#### *property* name *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

Unique identifier for this connector.

### *class* aix.ai_models.sources.OpenRouterSource(, timeout=30)

Bases: [`ModelSource`](aix.ai_models.base.html.md#aix.ai_models.base.ModelSource)

Discover models from OpenRouter’s public API.

OpenRouter maintains a comprehensive registry of models across providers.

#### discover_models()

Fetch models from OpenRouter API.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Model`](aix.ai_models.base.html.md#aix.ai_models.base.Model)]

```pycon
>>> source = OpenRouterSource()
>>> models = list(source.discover_models())
>>> len(models) > 0  # Should find many models
True
```

### *class* aix.ai_models.sources.ProviderAPISource(provider_name, , api_key=None, base_url='https://api.openai.com/v1', timeout=30)

Bases: [`ModelSource`](aix.ai_models.base.html.md#aix.ai_models.base.ModelSource)

Discover models from a provider’s API (OpenAI, Anthropic, etc.).

Generic source for providers with /v1/models endpoint.

#### discover_models()

Fetch models from provider API.

Requires valid API key for the provider.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Model`](aix.ai_models.base.html.md#aix.ai_models.base.Model)]
