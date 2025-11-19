# AIX Resources Summary

Comprehensive notes from the `aix_discussions.md` and `aix_py.md` files, focusing on resources, URLs, and implementation ideas for the AIX project.

---

## Current AIX Implementation (from aix_py.md)

### What Currently Exists
- **`chat_funcs`**: Mapping of model names → chat functions (as functools.partial)
- **`chat_models`**: Mapping of model names → metadata dicts (pricing, performance, context, provider)
- **`ai_models/`**: Sophisticated module for unified AI model management using Mapping/MutableMapping pattern
  - `base.py`: Core types (Model, ModelRegistry, Connector, ModelSource)
  - `sources.py`: Concrete implementations
  - `manager.py`: Unified ModelManager facade

### Design Pattern
- Uses functional approach with mapping interfaces
- Models accessible via attribute access (e.g., `chat_funcs.gpt_4o_mini`)
- Different providers have different interfaces normalized to common `prompt` parameter

---

## 🔑 Key Resources for Model Discovery & Information

### 1. **OpenRouter Models API** (RECOMMENDED PRIMARY SOURCE)
- **URL**: `https://openrouter.ai/api/v1/models`
- **Access**: Public, no auth required
- **What it provides**: 400+ models from 60+ providers
  - Model IDs in format: `"openai/gpt-4o"`, `"anthropic/claude-sonnet-4"`
  - Author/vendor info
  - Capabilities metadata
  - Per-model hosting endpoints
- **Why important**: Closest thing to a maintained, vendor-agnostic registry
- **Python example**:
```python
import requests
r = requests.get("https://openrouter.ai/api/v1/models", timeout=30)
data = r.json()
for m in data.get("data", []):
    print(m.get("id"), m.get("owned_by"))
```

### 2. **Provider-Native Model List Endpoints**
Official APIs from each provider:
- **OpenAI**: `GET /v1/models` (requires API key)
- **Anthropic**: `GET /v1/models` (requires API key)
- **Google Gemini**: `GET https://ai.google.dev/api/models`
- **Mistral**: Via their API docs
- **Cohere**: `GET /v2/models`

### 3. **Hugging Face Hub**
- **Purpose**: Discover/search open-source models
- **Python API**:
```python
from huggingface_hub import HfApi
api = HfApi()
models = api.list_models(filter="text-generation")
for m in models[:5]:
    print(m.modelId)
```
- **Search**: `https://huggingface.co/models`
- Filter by task, language, author, tags

### 4. **Ollama** (Local Models)
- **List local models**: `ollama list` (CLI) or `GET /api/tags` (API)
- **Note**: No global remote catalog API; models must be pulled first

---

## 💰 Pricing Information Sources

### Current Data in AIX
- Currently stores `price_per_million_tokens` in chat_models metadata
- Example: `chat_models['gpt-4o']` → `{'price_per_million_tokens': 5.0, ...}`

### External Pricing Resources
**Need to identify maintained pricing APIs/datasets** - This appears to be missing from the discussions file. Potential sources:
- OpenRouter API includes pricing data
- Provider-specific pricing pages (but not APIs)
- Consider scraping or maintaining a dataset

---

## 📊 Benchmark & Leaderboard Sources

### 1. **Comprehensive Benchmark Catalogs**

**Evidently AI - "250 LLM Benchmarks"**
- **URL**: https://www.evidentlyai.com/llm-evaluation-benchmarks-datasets
- **Updated**: July 31, 2025
- **Coverage**: 250+ benchmarks across reasoning, math, coding, safety, etc.
- **Format**: Links to datasets with metadata

**Our World in Data - AI Test Scores**
- **URL**: https://ourworldindata.org/grapher/test-scores-ai-capabilities-relative-human-performance
- **Format**: Downloadable CSV/JSON
- **Coverage**: Historical AI performance vs human baseline over time
- **Good for**: Longitudinal tracking

### 2. **Research Benchmark Repositories**

**BenchHub** (arXiv May 2025)
- **URL**: https://arxiv.org/abs/2506.00482
- **Coverage**: 303K questions over 38 benchmarks
- **Features**: Dynamic updates, customizable evaluations

**OneEval** (arXiv June 2025)
- **URL**: https://arxiv.org/abs/2506.12577
- **Focus**: Knowledge-intensive reasoning
- **Features**: Public dataset + leaderboard

**EvaLearn** (arXiv June 2025)
- **URL**: https://arxiv.org/abs/2506.02672
- **Focus**: Sequential learning capability
- **Coverage**: 648 problems

### 3. **Classic Benchmarks**

**MMLU** (Measuring Massive Multitask Language Understanding)
- **Coverage**: ~16K multiple-choice questions over 57 subjects
- **Human baseline**: ~89.8%
- **Reference**: https://en.wikipedia.org/wiki/MMLU

**Others**: GLUE, SuperGLUE, Big-Bench (BBH/BBEH), GSM8K, AQuA-RAT, Math23K

### 4. **Agentic/Long-Task Benchmarks**

**METR - Measuring AI's Ability to Complete Long Tasks**
- **URL**: https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/
- **Tracks**: Maximum task length doubling ~every 7 months

**AGIEval** (arXiv April 2023)
- **URL**: https://arxiv.org/abs/2304.06364
- **Focus**: Human-centric standardized exams (SAT, LSAT)

---

## 🛠️ Existing Python Packages for Model Management

### 1. **LiteLLM** (Most Popular)
- **PyPI**: `pip install litellm`
- **GitHub**: https://github.com/BerriAI/litellm
- **Downloads**: 1M+ monthly
- **Coverage**: 100+ LLM APIs
- **Features**:
  - Unified OpenAI-compatible interface
  - Cost tracking, guardrails, load balancing
  - Proxy server mode
  - Supports streaming, tool calling, multimodal
- **Usage**:
```python
from litellm import completion
response = completion(model="anthropic/claude-sonnet-4", messages=messages)
```

### 2. **any-llm** (Mozilla)
- **Approach**: Uses official provider SDKs (not re-implementations)
- **Philosophy**: More principled than LiteLLM
- **Release**: July 2025
- **Focus**: Leverage native SDKs, avoid compatibility drift
- **Docs**: https://huggingface.co/blog/mozilla-ai/introducing-any-llm

### 3. **LLM Registry** (Purpose-Built)
- **Package**: `llm-registry`
- **Purpose**: Centralized model management
- **Features**:
  - List models across providers
  - Rich metadata: context windows, features, costs
  - Filter by provider, source
  - JSON export
  - Local persistence (~/.llm-registry)
- **Usage**:
```python
from llm_registry import CapabilityRegistry
registry = CapabilityRegistry()
models = registry.get_models()
model = registry.get_model("gpt-4")
```
- **CLI**: `llmr list --provider openai`

### 4. **PyLLMs** (Kagi Search)
- **Minimal library** for unified LLM access
- **Coverage**: OpenAI, Anthropic, Google, AI21, Cohere, HuggingFace, Ollama
- **Usage**:
```python
import llms
model = llms.init('gpt-4')
result = model.complete("Hello world?")
print(result.text, result.meta)
```
- **Features**: Multi-model invocation, standardized metadata

### 5. **`llm` CLI** (Simon Willison)
- **GitHub**: https://github.com/simonw/llm
- **Type**: CLI tool + Python API
- **Plugin system** for providers
- **SQLite logging**
- **Aliases** for models

---

## 🌐 Gateway Services (Proxy/Aggregator)

### 1. **OpenRouter**
- **URL**: https://openrouter.ai
- **API**: OpenAI-compatible
- **Models**: 400+ from 60+ providers
- **Pricing**: Pass-through + small markup
- **Features**: Automatic failover, load balancing
- **Docs**: https://openrouter.ai/docs

### 2. **Portkey**
- **URL**: https://portkey.ai
- **Docs**: https://portkey.ai/docs/product/ai-gateway/universal-api
- **Features**: Fallbacks, routing, load balancing, analytics
- **API**: Both OpenAI and Anthropic compatible endpoints

### 3. **Cloudflare AI Gateway**
- **URL**: https://gateway.ai.cloudflare.com
- **Blog**: https://blog.cloudflare.com/ai-gateway-aug-2025-refresh/
- **Features**: Unified billing, secure key storage, dynamic routing, DLP
- **Format**: `{provider}/{model}` (e.g., `anthropic/claude-sonnet-4-5`)

### 4. **Vercel AI Gateway**
- **Docs**: https://ai-sdk.dev/providers/ai-sdk-providers/ai-gateway
- **Part of**: AI SDK 5.0.36+
- **Features**: Automatic provider detection, fallbacks

### 5. **Others**
- **Pydantic AI Gateway**: https://pydantic.dev/ai-gateway
- **Orq.ai**: https://docs.orq.ai/docs/proxy
- **Kong AI Gateway**: OpenAI-compatible middleware

---

## 📐 API Standards & Conventions

### OpenAI API Format (De Facto Standard)
- **Endpoints**:
  - `/v1/chat/completions` - Main chat
  - `/v1/embeddings` - Vector embeddings
  - `/v1/images/generations` - Image generation
  - `/v1/audio/transcriptions` - Speech-to-text
  - `/v1/models` - List models

### Structured Output Support
**Platforms with JSON Schema support**:
- **OpenAI**: https://platform.openai.com/docs/guides/structured-outputs
- **Anthropic Claude**: https://docs.anthropic.com/claude/docs/structured-output
- **Perplexity**: https://docs.perplexity.ai/guides/structured-outputs
- **LM Studio**: https://lmstudio.ai/docs/app/api/structured-output
- **OpenRouter**: https://openrouter.ai/docs/features/structured-outputs

---

## 🔌 Model Context Protocol (MCP)

### What is MCP?
- **Official Docs**: https://modelcontextprotocol.io
- **Protocol**: JSON-RPC 2.0 over various transports
- **Purpose**: Standardize how AI clients interact with tools/resources

### MCP vs OpenAPI
| Feature | OpenAPI | MCP |
|---------|---------|-----|
| Goal | REST APIs for devs | Capabilities for AI Agents |
| Schema | HTTP requests/responses | Single-object inputs/outputs |
| Communication | HTTP (stateless) | JSON-RPC (stateful) |
| Focus | Endpoints (CRUD) | Capabilities (descriptive) |

### MCP Features
- **Tools**: Callable actions
- **Resources**: Read-only data
- **Prompts**: Reusable workflows
- **Elicitation**: Server requests user input mid-operation
- **Sampling**: Server requests LLM completion
- **Security**: Client-enforced boundaries (Roots)

### Integration
- GitHub Copilot + MCP: https://hackernoon.com/ai-coding-assistants-learn-some-manners-with-read-only-mode
- Cursor supports MCP servers
- LiteLLM has beta MCP server integration

---

## 💡 Implementation Ideas & Architecture

### Recommended Architecture (Based on Your Style)

```python
# Core abstraction - Models as Mapping
class ModelStore(Mapping[str, 'ModelInfo']):
    """Discover and access model metadata"""
    def __getitem__(self, key) -> 'ModelInfo':
        # Support various access patterns:
        # models['openai/gpt-4o']  # Direct ID
        # models['provider:openai']  # Filter by provider
        # models[{'kind': 'chat', 'max_cost': 5.0}]  # Query
        ...
    
    def __iter__(self):
        """Iterate model IDs"""
        ...

# Inference as callable
class ChatModel:
    def __call__(
        self,
        messages: Iterable[dict],
        *,
        temperature: float = 1.0,
        stream: bool = False,
        **provider_specific
    ) -> dict | Iterable[dict]:
        ...

# Usage
models = ModelStore(backend='litellm')  # or 'native', 'openrouter'
gpt4 = models['openai/gpt-4o']
result = gpt4(messages=[{'role': 'user', 'content': 'Hello'}])
```

### Backend Strategy Options

**Option 1: Wrap LiteLLM** (Pragmatic)
- Use LiteLLM for provider logic (battle-tested)
- Add your Mapping interface on top
- Focus on UX and discovery

**Option 2: Pure Implementation** (Purist)
- Use `any-llm` approach (official SDKs)
- Build your own provider adapters
- More work but cleaner architecture

**Option 3: Hybrid**
- Use OpenRouter API for discovery
- Use provider SDKs for invocation
- Build unified Mapping interface

### Data Sources Strategy

1. **Model Discovery**: OpenRouter API (primary) + Provider APIs (augment)
2. **Benchmarks**: Evidently AI + OneEval + MMLU references
3. **Pricing**: OpenRouter + manual dataset (needs investigation)
4. **Local Models**: Ollama `/api/tags` + HuggingFace Hub search

---

## 🎯 Key Design Questions from Earlier

From our earlier discussion, these remain relevant:

1. **Streaming**: Return type change or always Iterable?
   - Recommendation: Always return Iterable that yields once if not streaming
   
2. **Tools/Functions**: Normalize schema or pass-through?
   - Recommendation: Start with pass-through, add normalization layer later

3. **Async**: Both sync/async or just async?
   - Recommendation: Start sync, add async wrappers

4. **Cost tracking**: Automatic or manual?
   - Recommendation: Optional automatic via wrapper/decorator

5. **Model discovery**: Static or real-time?
   - Recommendation: Cache with refresh, query OpenRouter periodically

---

## 📝 Additional Notes from Discussions

### AI Coding Assistants (Context)
- Many AI coding tools discussed (Cursor, Copilot, etc.)
- Relevant for understanding AI tool landscape
- MCP integration emerging as standard

### Memorization Concerns
- LLMs memorize ~1% of training data
- Privacy/copyright implications
- Companies use unlikelihood training, data filtering
- Custom data typically not used for base model training

### Text-to-JSON Tools
- Multiple LLMs support structured output
- OpenAI popularized JSON schema enforcement
- Consider this for AIX output handling

---

## 🚀 Next Steps Recommendations

1. **Start with OpenRouter**
   - Implement ModelStore using OpenRouter API
   - Get 400+ models immediately
   - Add provider-specific augmentation later

2. **Wrap LiteLLM Initially**
   - Proven, maintained, 1M+ users
   - Focus on your Mapping interface
   - Can swap backend later if needed

3. **Build Discovery Layer**
   - Cache OpenRouter model list
   - Add filtering/querying capabilities
   - Implement your Mapping patterns

4. **Add Metadata**
   - Integrate benchmark sources (Evidently AI, etc.)
   - Add pricing data (OpenRouter provides this)
   - Include capability flags

5. **Create Usage Layer**
   - Implement callable models
   - Add streaming support
   - Handle different parameter styles

---

## 📚 References

### Key Papers
- Small Language Models are the Future: https://arxiv.org/pdf/2506.02153
- Quantifying Memorization: https://arxiv.org/abs/2202.07646
- BenchHub: https://arxiv.org/abs/2506.00482
- OneEval: https://arxiv.org/abs/2506.12577

### Important Links
- Model Context Protocol: https://modelcontextprotocol.io
- OpenRouter: https://openrouter.ai
- LiteLLM: https://github.com/BerriAI/litellm
- Evidently AI Benchmarks: https://www.evidentlyai.com/llm-evaluation-benchmarks-datasets
- Hugging Face Hub: https://huggingface.co/models

---

*Document created from analysis of aix_discussions.md and aix_py.md files*
