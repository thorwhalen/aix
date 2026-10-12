"""OpenAI GenAI API functionality (legacy backend, via the optional ``oa`` package).

``oa`` is imported on the first call, never at ``import aix`` time: importing
``oa`` resolves an OpenAI API key as a side effect -- raising when none is
configured, and *prompting for one on stdin* in an interactive session -- so an
eager import made ``import aix`` crash or block for anyone who has ``oa``
installed but no key set (https://github.com/thorwhalen/aix/issues/54).
"""

from importlib.util import find_spec

name = "openai"
required_packages = ["oa"]

# The models this backend is registered for. (``oa``'s own, richer model table
# is not read at import time, for the reason in the module docstring.)
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


def _oa_is_installed() -> bool:
    """Whether ``oa`` can be imported, checked without importing it."""
    try:
        return find_spec("oa") is not None
    except (ImportError, ValueError):
        return False


if _oa_is_installed():

    def chat(prompt: str, *, model: str = "gpt-4o-mini", **kwargs):
        """Chat through ``oa``, importing it (and resolving its key) on first use."""
        from oa import chat as oa_chat

        return oa_chat(prompt, model=model, **kwargs)
