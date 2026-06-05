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

AIX resolves keys in this order: explicit `api_key=` argument → provider
environment variable (a project `.env` is picked up when `python-dotenv` is
installed) → the per-user AIX config store. Check what's discoverable with
`aix.check_keys()`, and see the
[Providing credentials](../README.md#providing-credentials) section for the full
precedence and per-provider env-var table. A missing key raises an actionable
`aix.MissingCredentialError` naming which key to set and where to get one.

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

### 07_image_generation.py
Text-to-image generation:
- Simple image generation
- High quality with DALL-E 3
- Different sizes and styles
- Multiple variations
- Creative prompts

```bash
python examples/07_image_generation.py
```

### 08_audio_operations.py
Audio operations (TTS and transcription):
- Text-to-speech with different voices
- Speed variations
- High quality TTS
- Audio transcription
- Language hints and timestamps
- Round-trip testing

```bash
python examples/08_audio_operations.py
```

### 09_multimodal_workflow.py
Combining multiple AI operations:
- Text → Image → Description workflows
- Text → Speech → Transcription → Analysis
- Story generation with multimedia
- Multilingual workflows
- Content creation pipelines

```bash
python examples/09_multimodal_workflow.py
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
