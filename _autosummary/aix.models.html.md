# aix.models

### aix.models *= ModelStore(0 models)*

User-friendly interface for model discovery and selection.

Provides a Mapping interface over the ModelManager with convenient
access patterns and integration with chat/embeddings functions.

### Examples

```pycon
>>> models = ModelStore()
>>> models.discover()  # Fetch available models
```

```pycon
>>> # List all models
>>> list(models)[:5]
['openai/gpt-4o', 'openai/gpt-4o-mini', ...]
```

```pycon
>>> # Get model info
>>> info = models['openai/gpt-4o']
>>> info.provider
'openai'
```

```pycon
>>> # Filter models
>>> openai_models = models.filter(provider='openai')
>>> local_models = models.filter(is_local=True)
```

```pycon
>>> # Use with chat
>>> from aix.chat import chat
>>> model = models['gpt-4o-mini']
>>> chat("Hello", model=model.id)
'Hello! How can I help you?'
```
