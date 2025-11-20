# AIX: Artificial Intelligence eXtensions

A clean, pythonic facade for common AI operations that abstracts away provider-specific details and complexities.

**Philosophy**: Make AI interactions as simple and intuitive as Python itself, while maintaining the power and flexibility needed for production use.

## Installation

```bash
pip install aix
```

## Quick Start

```python
from aix import chat, embeddings, prompt_func, models
from aix import generate_image, text_to_speech, transcribe

# Simple chat
response = chat("What is 2+2?")
print(response)  # "The answer is 4."

# Create prompt-based functions
translate = prompt_func("Translate to French: {text}")
result = translate(text="Hello world")
print(result)  # "Bonjour le monde"

# Get embeddings
vecs = list(embeddings(["hello", "world"]))
print(len(vecs))  # 2

# Generate images
image = generate_image("A serene mountain landscape")
image.save("landscape.png")

# Text to speech
audio = text_to_speech("Hello, world!")
audio.save("hello.mp3")

# Speech to text
text = transcribe("recording.mp3")

# Discover available models
models.discover()
print(list(models)[:5])  # ['openai/gpt-4o', 'openai/gpt-4o-mini', ...]
```

## Core Features

### 1. Chat Interface

Clean, provider-agnostic chat completions:

```python
from aix import chat

# Simple text prompt
response = chat("Explain quantum computing in one sentence")

# With specific model
response = chat("Hello!", model="gpt-4o-mini")

# With message history
messages = [
    {"role": "user", "content": "My name is Alice"},
    {"role": "assistant", "content": "Nice to meet you, Alice!"},
    {"role": "user", "content": "What's my name?"}
]
response = chat(messages, model="gpt-4o")

# Streaming responses
for chunk in chat("Count to 5", stream=True):
    print(chunk, end='', flush=True)

# Stateful conversations
from aix import chat_with_history

session = chat_with_history("You are a helpful math tutor")
response = session.send("What is 2+2?")
response = session.send("And if I add 3 to that?")  # Remembers context
```

### 2. Embeddings

Generate vector embeddings for semantic search and similarity:

```python
from aix import embeddings, embed, cosine_similarity, find_most_similar

# Batch embeddings
texts = ["cat", "dog", "bird"]
vecs = list(embeddings(texts))

# Single text
vec = embed("Hello, world!")

# Compute similarity
v1 = embed("cat")
v2 = embed("kitten")
similarity = cosine_similarity(v1, v2)  # High similarity

# Find most similar documents
query = "What is machine learning?"
docs = [
    "Machine learning is a type of AI",
    "Python is a programming language",
    "Neural networks are used in deep learning"
]
results = find_most_similar(query, docs, top_k=2)
# Returns: [('Machine learning is a type of AI', 0.95), ...]

# Caching for efficiency
from aix import EmbeddingCache

cache = EmbeddingCache()
v1 = cache.embed("hello")  # API call
v2 = cache.embed("hello")  # From cache
```

### 3. Prompt-Based Functions

Transform natural language prompts into reusable Python functions:

```python
from aix import prompt_func, prompt_to_text, prompt_to_json

# Simple text generation
summarize = prompt_func("Summarize this text: {text}")
summary = summarize(text="Long article...")

# Structured output
extract_person = prompt_func(
    "Extract person information from: {text}",
    output_schema={"name": str, "age": int, "email": str}
)
result = extract_person(text="Contact John at john@example.com. He is 30 years old.")
# Returns: {'name': 'John', 'age': 30, 'email': 'john@example.com'}

# Multiple parameters
compare = prompt_func(
    "Compare {item1} and {item2} in terms of {aspect}"
)
result = compare(
    item1="Python",
    item2="JavaScript",
    aspect="learning curve"
)

# Pre-built common functions
from aix import common_funcs

summary = common_funcs.summarize(text="Long article...")
keywords = common_funcs.extract_keywords(text="Article about AI and ML")
sentiment = common_funcs.sentiment(text="I love this product!")

# Create custom collections
from aix import PromptFuncs

my_funcs = PromptFuncs(model="gpt-4o")
my_funcs.add('analyze', "Analyze this code: {code}")
my_funcs.add('fix_bugs', "Fix bugs in: {code}")

result = my_funcs.analyze(code="def foo(): return bar")
```

### 4. Model Discovery & Selection

Discover and filter models across providers:

```python
from aix import models

# Discover available models
models.discover('openrouter')  # Fetch 400+ models from OpenRouter

# List all models
all_models = list(models)

# Get specific model info
info = models['openai/gpt-4o']
print(info.provider)  # 'openai'
print(info.context_size)  # 128000

# Filter models
openai_models = models.filter(provider='openai')
cheap_models = models.filter(
    custom_filter=lambda m: m.cost_per_token.get('input', 0) < 0.001
)
local_models = models.filter(is_local=True)

# Search models
results = models.search('gpt-4')
results = models.search('claude')

# Get recommendations
recommended = models.recommend(
    task='chat',
    max_cost_per_mtok=5.0,
    min_context_size=16000
)

# Use with chat
model = models['gpt-4o-mini']
response = chat("Hello", model=model.id)
```

### 5. Batch Operations

Process multiple requests efficiently:

```python
from aix import batch_chat, batch_embeddings, BatchProcessor

# Batch chat
prompts = ["What is 2+2?", "What is 3+3?", "What is 5+5?"]
results = list(batch_chat(prompts, batch_size=10, max_workers=5))

# Batch embeddings
texts = ["hello", "world", "foo", "bar"] * 100
vectors = list(batch_embeddings(
    texts,
    batch_size=20,
    show_progress=True
))

# Generic batch processing
from aix import batch_process

def analyze(text):
    return chat(f"Analyze sentiment: {text}")

texts = ["I love it!", "It's okay", "Terrible"]
results = list(batch_process(
    texts,
    analyze,
    batch_size=5,
    retry_attempts=3
))

# Stateful batch processor
processor = BatchProcessor(show_progress=True)
results = processor.process_chats(prompts)
processor.save_results("output.json")
```

### 6. Image Generation

Generate images from text descriptions:

```python
from aix import generate_image, generate_images

# Simple image generation
image = generate_image("A serene mountain landscape at sunset")
image.save("landscape.png")

# High quality with DALL-E 3
image = generate_image(
    "Abstract art with vibrant colors",
    model="dall-e-3",
    quality="hd",
    style="vivid"
)

# Generate multiple variations
images = generate_images(
    "A cute robot waving hello",
    n=3,
    size="512x512"
)
for i, img in enumerate(images):
    img.save(f"robot_{i}.png")

# Edit existing images
from aix import edit_image

edited = edit_image(
    "photo.jpg",
    "Add a rainbow in the sky",
    mask_path="sky_mask.png"
)

# Create variations
from aix import create_variation

variations = create_variation("original.png", n=3)
```

### 7. Audio Operations

Text-to-speech and speech-to-text:

```python
from aix import text_to_speech, transcribe, transcribe_with_timestamps

# Text to speech
audio = text_to_speech("Hello, world!")
audio.save("hello.mp3")

# Different voices
audio = text_to_speech(
    "This is a test",
    voice="nova",
    speed=1.2
)

# Transcribe audio
text = transcribe("recording.mp3")
print(text)  # "This is the transcribed text"

# With language hint
text = transcribe("spanish_audio.mp3", language="es")

# Detailed transcription with timestamps
result = transcribe_with_timestamps("lecture.mp3")
for segment in result.segments:
    print(f"[{segment['start']:.2f}] {segment['text']}")

# Translate audio to English
from aix import translate_audio

english_text = translate_audio("spanish_audio.mp3")
```

### 8. Video Generation (Coming Soon)

Video generation with provider-specific implementations:

```python
from aix import generate_video, get_video_providers

# Check available providers
providers = get_video_providers()
print(providers)  # ['runway', 'pika', ...]

# Generate video (requires provider setup)
video = generate_video(
    "A cat walking through a garden",
    duration=5,
    resolution="1920x1080"
)
video.save("cat_video.mp4")

# Animate static image
from aix import animate_image_to_video

video = animate_image_to_video(
    "landscape.jpg",
    prompt="Gentle camera pan across the scene"
)
```

**Note**: Video generation requires additional provider setup (Runway, Pika, etc.) and API keys.

## OpenRouter Integration

[OpenRouter](https://openrouter.ai) provides a single API key for 400+ models from 60+ providers. This is recommended for:

- **Getting started quickly** - One key vs. managing many
- **Experimenting** - Easy access to models from OpenAI, Anthropic, Google, etc.
- **Production flexibility** - Switch providers without code changes

### Setup

1. Get API key from https://openrouter.ai
2. Set environment variable:
   ```bash
   export OPENROUTER_API_KEY=your-key-here
   ```
3. Use OpenRouter models:
   ```python
   # Prefix models with 'openrouter/'
   chat("Hello", model="openrouter/openai/gpt-4o")
   chat("Hello", model="openrouter/anthropic/claude-3.5-sonnet")

   # Discover available models
   models.discover('openrouter')
   ```

All standard AIX features work with OpenRouter models.

## Architecture

AIX follows the **i2mint philosophy** of clean, functional interfaces:

- **Mapping interfaces** for collections (models, registries)
- **Functional approach** over verbose OOP
- **Lazy evaluation** via generators where appropriate
- **Protocol-based design** for flexibility

### Backend: LiteLLM

AIX uses [LiteLLM](https://github.com/BerriAI/litellm) as the backend for provider interactions, but wraps it in clean, pythonic interfaces. Users never need to interact with LiteLLM directly.

**Supported Providers** (via LiteLLM):
- OpenAI
- Anthropic (Claude)
- Google (Gemini)
- Mistral
- Cohere
- And 100+ more

## Design Patterns

### From `oa` (OpenAI facade)

AIX builds on patterns from the `oa` package:
- Simple chat interface with smart defaults
- Template-based function creation
- Structured output support
- Batch processing capabilities

### From `i2` (Signature manipulation)

- Clean function signatures
- Flexible parameter handling
- Decorator-based composition

### From `dol` (Storage abstraction)

- Mapping-based interfaces
- Persistent storage options
- Cache management

## Advanced Usage

### Custom Model Sources

```python
from aix.ai_models import ModelManager, OpenRouterSource

manager = ModelManager()
source = OpenRouterSource()
models = manager.discover_from_source('openrouter')
```

### Connector-Specific Metadata

```python
# Get provider-specific parameters
metadata = models.get_connector_metadata('openai/gpt-4o', 'openai')
# Use with native SDK: openai.ChatCompletion.create(**metadata, messages=[...])
```

### Error Handling

```python
from aix import chat

try:
    response = chat("Hello", model="nonexistent-model")
except Exception as e:
    print(f"Error: {e}")
```

## Backward Compatibility

The legacy `chat_funcs` and `chat_models` interfaces are still available:

```python
from aix import chat_funcs, chat_models

# Old style (still works)
list(chat_funcs)  # ['gpt-4o', 'gpt-4o-mini', ...]
response = chat_funcs.gpt_4o("Hello")

# Model metadata
info = chat_models['gpt-4o']
# {'price_per_million_tokens': 5.0, 'provider': 'openai', ...}
```

However, the new interfaces (`chat`, `embeddings`, `prompt_func`, `models`) are recommended for new code.

## Examples

### Semantic Search

```python
from aix import embed, cosine_similarity

# Build document index
docs = ["AI is the future", "Python is great", "Machine learning works"]
doc_vecs = [embed(doc) for doc in docs]

# Search
query_vec = embed("artificial intelligence")
similarities = [cosine_similarity(query_vec, dv) for dv in doc_vecs]
best_match = docs[similarities.index(max(similarities))]
print(best_match)  # "AI is the future"
```

### Data Extraction Pipeline

```python
from aix import prompt_func, batch_process

# Define extraction function
extract = prompt_func(
    "Extract product info from: {text}",
    output_schema={"name": str, "price": float, "category": str}
)

# Process many product descriptions
descriptions = [...]  # Your data
results = list(batch_process(
    descriptions,
    lambda d: extract(text=d),
    batch_size=10,
    show_progress=True
))
```

### Multi-Model Comparison

```python
from aix import chat

prompt = "Explain quantum computing in one sentence"

# Try different models
for model_id in ['gpt-4o-mini', 'claude-sonnet-4', 'gemini-1.5-flash']:
    response = chat(prompt, model=model_id)
    print(f"{model_id}: {response}")
```

## Documentation

- **API Reference**: See docstrings in each module
- **Examples**: Check `examples/` directory
- **GitHub**: https://github.com/thorwhalen/aix

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## License

Apache 2.0

## Credits

Built with:
- [LiteLLM](https://github.com/BerriAI/litellm) - Multi-provider backend
- [i2](https://github.com/i2mint/i2) - Signature manipulation utilities
- [dol](https://github.com/i2mint/dol) - Storage abstraction patterns
- [oa](https://github.com/thorwhalen/oa) - OpenAI facade inspiration

## What AIX is NOT

AIX is NOT an AI agent framework. For that, see the separate `aw` package. AIX focuses on foundational AI operations (chat, embeddings, etc.) that can be used to build agents and other applications.
