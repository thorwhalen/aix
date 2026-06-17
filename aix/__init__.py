"""AIX: Artificial Intelligence eXtensions

A clean, pythonic facade for common AI operations that abstracts away
provider-specific details and complexities.

Quick Start:
    >>> from aix import chat, embeddings, prompt_func, models

    # Simple chat
    >>> response = chat("What is 2+2?")  # doctest: +SKIP
    'The answer is 4.'

    # Create prompt-based functions
    >>> translate = prompt_func("Translate to French: {text}")
    >>> translate(text="Hello world")  # doctest: +SKIP
    'Bonjour le monde'

    # Get embeddings
    >>> vecs = list(embeddings(["hello", "world"]))  # doctest: +SKIP
    >>> len(vecs)  # doctest: +SKIP
    2

    # Discover models
    >>> models.discover()  # doctest: +SKIP
    >>> list(models)[:5]  # doctest: +SKIP
    ['openai/gpt-4o', 'openai/gpt-4o-mini', ...]

Main Features:
    - chat(): Simple chat interface across providers
    - embeddings(): Vector embeddings for text
    - prompt_func(): Create functions from prompt templates
    - models: Model discovery and selection
    - generate_image(): Text-to-image generation
    - text_to_speech(), transcribe(): Audio operations
    - generate_video(): Text-to-video generation (provider-dependent)
    - Batch operations for efficiency
    - Clean, i2mint-style Mapping interfaces

Backends:
    - Uses LiteLLM for provider interactions
    - Supports OpenAI, Anthropic, Google, and 100+ models
    - OpenRouter integration for multi-provider access

For detailed documentation, see: https://github.com/thorwhalen/aix
"""

# Configuration (single source of truth for per-function defaults)
from aix import config
from aix.config import (
    AixConfig,
    get_config,
    set_config,
    configure,
    using,
    load_config,
    resolve_model,
)

# Credentials (unified, discoverable API-key resolution)
from aix.credentials import (
    resolve_api_key,
    check_requirements,
    check_keys,
    MissingCredentialError,
    PROVIDER_ENV_VARS,
)

# Core interfaces (new clean API)
from aix.chat import chat, ask, chat_with_history, ChatSession
from aix.embeddings import (
    embeddings,
    embed,
    cosine_similarity,
    find_most_similar,
    EmbeddingCache,
    batched_embeddings,
    cached_embeddings,
    iter_batches,
    text_cache_key,
    truncate_segment,
)
from aix.prompts import (
    prompt_func,
    prompt_to_text,
    prompt_to_json,
    PromptFuncs,
    common_funcs,
    constrained_answer,
)
from aix.models import (
    models,
    ModelStore,
    discover_available_models,
    get_model_info,
    find_models,
)
from aix.batches import (
    batch_chat,
    batch_embeddings,
    batch_process,
    BatchProcessor,
)
from aix.image import (
    generate_image,
    generate_images,
    edit_image,
    create_variation,
    GeneratedImage,
)
from aix.audio import (
    text_to_speech,
    transcribe,
    transcribe_with_timestamps,
    translate_audio,
    GeneratedAudio,
    TranscriptionResult,
)
from aix.video import (
    generate_video,
    animate_image as animate_image_to_video,
    extend_video,
    GeneratedVideo,
    get_available_providers as get_video_providers,
)
from aix.vision import (
    describe_image,
    to_image_content,
)

# Legacy interfaces (for backward compatibility)
from aix.gen_ai import chat_models, chat_funcs

# Version info
__version__ = "0.1.0"

# Public API
__all__ = [
    # Configuration
    "config",
    "AixConfig",
    "get_config",
    "set_config",
    "configure",
    "using",
    "load_config",
    "resolve_model",
    # Credentials
    "resolve_api_key",
    "check_requirements",
    "check_keys",
    "MissingCredentialError",
    "PROVIDER_ENV_VARS",
    # Core chat
    "chat",
    "ask",
    "chat_with_history",
    "ChatSession",
    # Embeddings
    "embeddings",
    "embed",
    "cosine_similarity",
    "find_most_similar",
    "EmbeddingCache",
    # Prompts
    "prompt_func",
    "prompt_to_text",
    "prompt_to_json",
    "PromptFuncs",
    "common_funcs",
    "constrained_answer",
    # Models
    "models",
    "ModelStore",
    "discover_available_models",
    "get_model_info",
    "find_models",
    # Batches
    "batch_chat",
    "batch_embeddings",
    "batch_process",
    "BatchProcessor",
    # Image
    "generate_image",
    "generate_images",
    "edit_image",
    "create_variation",
    "GeneratedImage",
    # Audio
    "text_to_speech",
    "transcribe",
    "transcribe_with_timestamps",
    "translate_audio",
    "GeneratedAudio",
    "TranscriptionResult",
    # Video
    "generate_video",
    "animate_image_to_video",
    "extend_video",
    "GeneratedVideo",
    "get_video_providers",
    # Vision (image -> text)
    "describe_image",
    "to_image_content",
    # Legacy
    "chat_models",
    "chat_funcs",
]
