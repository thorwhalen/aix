# AIX Examples

This directory contains practical examples demonstrating AIX functionality.

## Prerequisites

Make sure you have AIX installed and necessary API keys set:

```bash
pip install aix

# Set API keys (choose the ones you need)
export OPENAI_API_KEY=your-key-here
export ANTHROPIC_API_KEY=your-key-here
export OPENROUTER_API_KEY=your-key-here
```

## Examples

### 01_simple_chat.py
Basic chat functionality demonstrating:
- Simple questions
- Model selection
- Temperature control
- Message history
- Streaming responses

```bash
python examples/01_simple_chat.py
```

### 02_stateful_chat.py
Maintaining conversation history with ChatSession:
- System prompts
- Multi-turn conversations
- History management
- Context preservation

```bash
python examples/02_stateful_chat.py
```

### 03_embeddings.py
Vector embeddings and semantic search:
- Batch embeddings
- Single text embedding
- Cosine similarity
- Finding similar documents
- Pre-computed vectors

```bash
python examples/03_embeddings.py
```

### 04_prompt_functions.py
Creating reusable functions from prompts:
- Simple text functions
- Multiple parameters
- Structured output (JSON)
- Pre-built common functions
- Custom function collections

```bash
python examples/04_prompt_functions.py
```

### 05_batch_operations.py
Efficient batch processing:
- Batch chat
- Batch embeddings
- Generic batch processing
- BatchProcessor class
- Retry logic

```bash
python examples/05_batch_operations.py
```

### 06_model_discovery.py
Discovering and selecting models:
- Model discovery from sources
- Filtering and searching
- Getting recommendations
- Comparing models
- Using models with chat

```bash
python examples/06_model_discovery.py
```

## Running All Examples

```bash
# Run all examples in sequence
for f in examples/*.py; do
    echo "Running $f..."
    python "$f"
    echo "---"
done
```

## Common Patterns

### Error Handling

```python
from aix import chat

try:
    response = chat("Hello", model="gpt-4o-mini")
except Exception as e:
    print(f"Error: {e}")
```

### Custom Configuration

```python
from aix import chat

response = chat(
    "Hello",
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=100
)
```

### Using OpenRouter

OpenRouter provides access to 400+ models with a single API key:

```python
from aix import chat, models

# Set your OpenRouter API key
# export OPENROUTER_API_KEY=your-key-here

# Discover models
models.discover('openrouter')

# Use any model
chat("Hello", model="openrouter/openai/gpt-4o")
chat("Hello", model="openrouter/anthropic/claude-3.5-sonnet")
chat("Hello", model="openrouter/google/gemini-pro")
```

## Need Help?

- **Documentation**: See the main README.md
- **API Reference**: Check docstrings in source code
- **Issues**: https://github.com/thorwhalen/aix/issues
