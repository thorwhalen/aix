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
    "o1": {},
    "o1-preview": {},
    "o1-mini": {},
    "gpt-4o": {},
    "gpt-4o-mini": {},
}


def _oa_is_installed() -> bool:
    """Whether ``oa`` is installed, checked without importing it.

    A namespace package (a stray ``oa/`` folder without ``__init__.py`` on the
    path) has no ``origin`` and does not count.
    """
    try:
        spec = find_spec("oa")
    except (ImportError, ValueError):
        return False
    return spec is not None and spec.origin not in (None, "namespace")


if _oa_is_installed():

    def chat(*args, **kwargs):
        """Chat through ``oa.chat`` (same arguments), importing ``oa`` on first use.

        Everything is passed through, so ``oa``'s own defaults (e.g. its
        configured default model) and options (``messages=``...) still apply.
        """
        try:
            from oa import chat as oa_chat
        except ImportError as e:
            raise ImportError(
                f"The {name!r} legacy chat backend requires missing packages: "
                f"{required_packages}. Please install them and try again."
            ) from e
        return oa_chat(*args, **kwargs)
