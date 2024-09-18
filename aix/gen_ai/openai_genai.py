"""OpenAI GenAI API functionality."""

from contextlib import suppress

name = "openai"
required_packages = ["oa"]

# Will be overridden by the actual configuration from oa if oa installed
# (Here just to have a map between model and service)
chat_models = {
    "gpt-4": {},
    "gpt-4-32k": {},
    "gpt-4-turbo": {},
    "gpt-3.5-turbo": {},
    "o1-preview": {},
    "o1-mini": {},
    "gpt-4o": {},
    "gpt-4o-mini": {},
}

with suppress(ModuleNotFoundError, ImportError):
    from oa import chat
    from oa.util import chat_models
