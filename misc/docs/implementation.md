# AI Agent Prompt: Develop the AIX Package

## CRITICAL: Read These Resources First

Before starting any implementation, you MUST thoroughly read:

1. **`misc/docs/aix_resources_summary.md`** - Complete resource guide for AIX development
2. **The entire `oa` codebase** at https://github.com/thorwhalen/oa - This is our reference implementation
3. **i2 package** at https://github.com/i2mint/i2 - Signature manipulation and function wrapping utilities
4. **dol package** at https://github.com/i2mint/dol - Dictionary-oriented storage abstraction patterns

These resources contain the architectural patterns, design philosophy, and concrete examples you need to follow.

---

## Project Overview: AIX (Artificial Intelligence eXtensions)

**Goal**: Create a clean, pythonic facade for common AI operations that abstracts away provider-specific details, interfaces, and complexities.

**Philosophy**: 
- **Facade over Implementation**: Use LiteLLM as the backend for provider interactions, but wrap it in clean, i2mint-style interfaces
- **Discovery-Friendly**: Make it easy to search, view, and select AI models
- **Provider-Agnostic**: Users shouldn't need to know or care which provider they're using unless they want to
- **OpenRouter-Friendly**: Document OpenRouter as a convenient single-key solution for multi-provider access

**What AIX is NOT**: This is NOT an AI agent framework (that's the separate `aw` package). We're building foundational tools for AI interactions.

---

## Architecture Foundations

### Backend: LiteLLM
```python
# LiteLLM provides the multi-provider abstraction
from litellm import completion, embedding

# We wrap this in clean, Mapping-based interfaces
# User never calls litellm directly in typical usage
```

### Design Patterns (from i2 and dol)

**Use these i2mint patterns throughout**:
1. **Mapping/MutableMapping interfaces** for collections (models, providers, etc.)
2. **Functional approach** over OOP where appropriate
3. **i2.Sig** for signature manipulation when needed
4. **dol patterns** for any storage/caching needs
5. **Lazy evaluation** via generators where appropriate
6. **Protocol-based design** for flexibility

**Example of the i2mint style**:
```python
# Good - Mapping interface
models = ModelStore()
gpt4 = models['openai/gpt-4o']
result = gpt4("Hello")

# Also good - Attribute access for convenience
result = models.gpt_4o("Hello")

# Bad - Verbose OOP
client = ModelClient(provider='openai')
model = client.get_model('gpt-4o')
result = model.chat(messages=[...])
```

---

## Key Interfaces to Implement (from oa)

After reading the entire `oa` codebase, you should implement AIX equivalents of these core functions. Here's the essential API surface (names may evolve, but functionality is key):

### 1. **Chat Interface** (from oa.openai_functions)

```python
def chat(
    prompt: str | Iterable[dict],
    *,
    model: str = None,
    temperature: float = 1.0,
    max_tokens: int = None,
    stream: bool = False,
    **kwargs
) -> str | Iterable[str]:
    """
    Simple chat completion.
    
    Examples:
        >>> chat("What is 2+2?")
        'The answer is 4.'
        
        >>> chat([{"role": "user", "content": "Hello"}], model="gpt-4o")
        'Hello! How can I help you today?'
    """
```

**Note**: Look at how `oa` handles both string prompts and message lists. Replicate that flexibility.

### 2. **Prompt-Based Function Creation** (IMPORTANT - Read Carefully)

In `oa`, there are currently TWO functions:
- `prompt_function(prompt, ...) -> callable` - Returns function that generates text
- `prompt_json_function(prompt, output_schema, ...) -> callable` - Returns function that generates structured JSON

**Your Task**: I want you to **unify these concepts** with a better name. Here's my thinking:

**Option A: Single Unified Function**
```python
def prompt_based_func(
    prompt: str,
    *,
    output_schema: dict | type = None,  # If provided, enables structured output
    model: str = None,
    **kwargs
) -> callable:
    """
    Create a callable function from a prompt.
    
    Without output_schema: Returns text
    With output_schema: Returns structured data
    
    Examples:
        # Text generation
        >>> summarize = prompt_based_func("Summarize: {text}")
        >>> summarize(text="Long article...")
        'Brief summary...'
        
        # Structured output
        >>> extract = prompt_based_func(
        ...     "Extract person info from: {text}",
        ...     output_schema={"name": str, "age": int}
        ... )
        >>> extract(text="John is 30 years old")
        {'name': 'John', 'age': 30}
    """
```

**Option B: Separate + Unified**
```python
# Keep separate for clarity
def prompt_to_text_func(prompt, **kwargs) -> callable: ...
def prompt_to_json_func(prompt, output_schema, **kwargs) -> callable: ...

# But provide unified interface
def prompt_based_func(prompt, output_schema=None, **kwargs):
    """Dispatches to appropriate implementation based on output_schema"""
    if output_schema is None:
        return prompt_to_text_func(prompt, **kwargs)
    else:
        return prompt_to_json_func(prompt, output_schema, **kwargs)
```

**Your Decision**: Choose what makes most sense after studying `oa.openai_functions.prompt_function` and `oa.openai_functions.prompt_json_function`. Consider:
- API simplicity vs explicitness
- Type hints and IDE support
- Common use cases from oa examples

Whatever you choose, document it well and provide doctests.

### 3. **Embeddings Interface** (from oa.openai_functions)

```python
def embeddings(
    segments: Iterable[str],
    *,
    model: str = None,
    **kwargs
) -> Iterable[Sequence[float]]:
    """
    Get embeddings for text segments.
    
    Examples:
        >>> vecs = list(embeddings(["hello", "world"]))
        >>> len(vecs)
        2
        >>> len(vecs[0])  # dimension
        1536
    """
```

### 4. **Model Discovery & Selection**

This is NEW compared to oa - build on your existing `ai_models/` module:

```python
# Primary interface
models = ModelStore()  # Uses OpenRouter + provider APIs for discovery

# Access patterns
list(models)  # All model IDs
models['openai/gpt-4o']  # Get specific model metadata
models[{'task': 'chat', 'max_cost_per_mtok': 5}]  # Query by criteria

# Convenience attributes
models.gpt_4o  # Tab-completable access

# Get model info
info = models.get_model_info('openai/gpt-4o')
# Returns: {'provider': 'openai', 'pricing': {...}, 'benchmarks': {...}, ...}
```

**Integration with chat/embeddings**:
```python
# User can pass model string or model object
chat("Hello", model="gpt-4o")  # String
chat("Hello", model=models.gpt_4o)  # Object (should work the same)
```

### 5. **Batch Operations** (from oa.batches and oa.batch_embeddings)

Study `oa/batches.py` and `oa/batch_embeddings.py`. Check if LiteLLM offers batch operation facades. If yes:

```python
def batch_chat(prompts: Iterable[str], **kwargs) -> Iterable[str]:
    """Process multiple prompts in batch for efficiency"""
    
def batch_embeddings(segments: Iterable[str], **kwargs) -> Iterable[Sequence[float]]:
    """Get embeddings for many segments efficiently"""
```

If LiteLLM doesn't offer this, implement a simple wrapper that chunks and processes efficiently.

### 6. **Chat Parsing** (from oa.chats)

Study `oa/chats.py` carefully - it's a nice interface for parsing shared chat URLs.

**Challenge**: Check if LiteLLM offers any facade for this. Likely not, since it's provider-specific.

**Options**:
- If LiteLLM doesn't support: Implement basic version for OpenAI shared chats (most common)
- Document as "experimental" or "limited provider support"
- Consider making it pluggable (provider-specific implementations)

```python
def parse_chat(url: str) -> list[dict]:
    """
    Parse a shared chat URL into structured conversation.
    
    Examples:
        >>> messages = parse_chat("https://chat.openai.com/share/...")
        >>> messages[0]
        {'role': 'user', 'content': 'Hello'}
    """
```

---

## Key Functions from oa to Study

When reading the `oa` codebase, pay special attention to these files and functions:

### openai_functions.py (Core API)
- `chat()` - Main chat interface
- `prompt_function()` - Template-based function creation
- `prompt_json_function()` - Structured output variant
- `embeddings()` - Vector embeddings
- `dflt_configs` - Default configuration handling
- Error handling patterns

### batches.py
- Batch operation patterns
- Cost optimization strategies
- Progress tracking

### batch_embeddings.py  
- Embedding batch processing
- Chunking strategies

### chats.py
- Chat URL parsing
- Message structure handling

### util.py
- Helper functions
- Common patterns
- Signature manipulation with i2

### openai_specs.py
- OpenAPI spec generation
- Schema handling
- Model definitions

**Look for patterns like**:
- How defaults cascade (global → function-specific)
- How different input types are normalized
- Error handling and retries
- Use of `i2.Sig` for signature manipulation
- Doctest examples

---

## Implementation Guidelines

### 1. Project Structure

```
aix/
├── __init__.py           # Public API exports
├── base.py              # Core abstractions (if needed beyond ai_models/)
├── chat.py              # Chat interface
├── prompts.py           # Prompt-based function creation
├── embeddings.py        # Embeddings interface
├── models.py            # Model discovery (build on ai_models/)
├── batches.py           # Batch operations (if LiteLLM supports)
├── util.py              # Utilities
├── ai_models/           # Existing model management (keep and extend)
│   ├── __init__.py
│   ├── base.py
│   ├── sources.py
│   ├── manager.py
│   └── ...
└── misc/
    └── docs/
        └── aix_resources_summary.md  # Your reference
```

### 2. LiteLLM Integration Pattern

**DON'T expose LiteLLM directly**:
```python
# Bad
from litellm import completion
user_result = completion(...)  # User calls litellm directly
```

**DO wrap it cleanly**:
```python
# Good - in aix internals
from litellm import completion as _litellm_completion

def chat(prompt, **kwargs):
    # Normalize inputs
    messages = _normalize_prompt(prompt)
    # Map aix kwargs to litellm format
    litellm_kwargs = _map_kwargs(kwargs)
    # Call litellm
    response = _litellm_completion(messages=messages, **litellm_kwargs)
    # Normalize output
    return _extract_text(response)
```

### 3. Model Selection Integration

```python
# In chat.py or similar
def chat(prompt, model=None, **kwargs):
    if model is None:
        model = get_default_model()
    
    # Support both string and ModelInfo objects
    if isinstance(model, str):
        model_id = model
    else:
        model_id = getattr(model, 'id', str(model))
    
    # Now use model_id with litellm
    ...
```

### 4. Configuration Management

Follow the `oa` pattern of cascading defaults:
```python
# Global defaults
DFLT_CHAT_MODEL = 'gpt-4o-mini'
DFLT_TEMPERATURE = 1.0

# Function-specific defaults via kwargs
def chat(prompt, model=None, temperature=None, **kwargs):
    model = model or DFLT_CHAT_MODEL
    temperature = temperature if temperature is not None else DFLT_TEMPERATURE
    ...
```

### 5. Signature Design with i2

Study how `oa` uses `i2.Sig` for flexible signatures. Example pattern:
```python
from i2 import Sig

@Sig.replace_kwargs_using_other_func(some_function)
def my_function(...):
    """Inherits parameter docs and validation"""
```

### 6. Error Handling

Look at `oa`'s error handling - replicate the pattern:
- Catch provider-specific errors
- Re-raise as generic AIX errors
- Include helpful context

### 7. Documentation

Every function needs:
- Clear docstring
- **At least one doctest** (prefer simple, runnable examples)
- Type hints
- Parameter descriptions

Example:
```python
def chat(
    prompt: str,
    *,
    model: str = None,
    temperature: float = 1.0,
) -> str:
    """
    Send a prompt and get a text response.
    
    Args:
        prompt: The text prompt to send
        model: Model identifier (e.g., 'gpt-4o', 'claude-sonnet-4')
        temperature: Sampling temperature (0.0 = deterministic, 2.0 = creative)
    
    Returns:
        Generated text response
        
    Examples:
        >>> chat("What is 2+2?")  # doctest: +SKIP
        'The answer is 4.'
    """
```

---

## OpenRouter Documentation

In the README and/or docs, explain OpenRouter clearly:

```markdown
### Using OpenRouter for Simplified Access

[OpenRouter](https://openrouter.ai) provides a single API key for 400+ models 
from 60+ providers. This is recommended for:

- **Getting started quickly** - One key vs. managing many
- **Experimenting** - Easy access to models from OpenAI, Anthropic, Google, etc.
- **Production flexibility** - Switch providers without code changes

To use with AIX:
1. Get API key from https://openrouter.ai
2. Set environment: `export OPENROUTER_API_KEY=your-key`
3. Prefix models: `chat("Hello", model="openrouter/gpt-4o")`

All standard AIX features work with OpenRouter models.
```

---

## Testing Strategy

Look at `oa`'s testing patterns. For AIX:

1. **Doctests**: Simple examples in docstrings
2. **Unit tests**: Test internal utilities
3. **Integration tests**: Test LiteLLM integration (may require API keys)
4. **Mock tests**: Mock LiteLLM responses for CI/CD

Example test structure:
```python
# In tests/test_chat.py
def test_chat_with_string_prompt():
    result = chat("Say 'test'", model="gpt-4o-mini")
    assert isinstance(result, str)
    assert len(result) > 0

def test_chat_with_messages():
    messages = [{"role": "user", "content": "Hello"}]
    result = chat(messages)
    assert isinstance(result, str)
```

---

## Specific Implementation Tasks

After reading all the resources, implement in this order:

1. **Core Chat Interface** (`chat.py`)
   - Study `oa.openai_functions.chat`
   - Wrap LiteLLM's completion
   - Support string and message list inputs
   - Handle streaming

2. **Embeddings** (`embeddings.py`)
   - Study `oa.openai_functions.embeddings`
   - Wrap LiteLLM's embedding
   - Return consistent format

3. **Prompt-Based Functions** (`prompts.py`)
   - Study both `prompt_function` and `prompt_json_function` from oa
   - Design unified interface (your call on naming/structure)
   - Support structured output via LiteLLM
   - Include i2.Sig manipulation if helpful

4. **Model Discovery** (`models.py`)
   - Extend existing `ai_models/` module
   - Implement ModelStore as Mapping
   - Integrate OpenRouter API for model listing
   - Add filtering/querying capabilities

5. **Batch Operations** (`batches.py`)
   - Check LiteLLM documentation for batch support
   - Study `oa.batches` and `oa.batch_embeddings`
   - Implement if LiteLLM supports, otherwise simple wrapper

6. **Chat Parsing** (if feasible)
   - Study `oa.chats`
   - Check LiteLLM capabilities
   - Implement basic version or mark as future work

7. **Documentation**
   - README with quickstart
   - API reference
   - OpenRouter integration guide

---

## What Success Looks Like

A user should be able to:

```python
from aix import chat, embeddings, prompt_based_func, models

# Simple chat
response = chat("Explain quantum computing in one sentence")

# Use different models easily
response = chat("Same question", model="claude-sonnet-4")

# Create prompt-based functions
translate = prompt_based_func("Translate to French: {text}")
result = translate(text="Hello world")

# Structured output
extract = prompt_based_func(
    "Extract: {text}",
    output_schema={"name": str, "age": int}
)
data = extract(text="Alice is 30")  # Returns dict

# Get embeddings
vecs = list(embeddings(["hello", "world"]))

# Discover models
list(models)[:5]  # Show first 5
models.gpt_4o  # Get model info
models[{'task': 'chat'}]  # Filter by task
```

---

## Questions to Ask Yourself While Coding

1. **Is this using i2/dol patterns?** (Mapping interfaces, functional style)
2. **Does this match the philosophy from oa?** (Clean, simple, well-documented)
3. **Am I exposing LiteLLM directly?** (No - wrap it)
4. **Does this need a doctest?** (Yes, unless it requires complex setup)
5. **Is this pythonic?** (Duck typing, generators, context managers where appropriate)
6. **Would Thor approve?** (Check against the coding standards in userPreferences)

---

## Final Reminders

- **READ EVERYTHING FIRST**: Resources summary, oa codebase, i2, dol
- **Don't reinvent**: Use LiteLLM for provider logic
- **Do innovate**: Build the cleanest, most pythonic API on top
- **Stay focused**: AI operations, NOT AI agents (that's `aw`)
- **Follow patterns**: i2mint style throughout
- **Document well**: Doctests, examples, clear docs
- **Test thoroughly**: Unit, integration, and doctest

Good luck! Build something beautiful. 🚀
