"""Core chat interface for AIX.

A clean, pythonic facade for chat completions across multiple AI providers.
Uses LiteLLM as the backend but provides a simpler, more intuitive interface.

Examples:
    Simple text prompt:
    >>> from aix.chat import chat
    >>> response = chat("What is 2+2?")  # doctest: +SKIP
    >>> print(response)  # doctest: +SKIP
    'The answer is 4.'

    With specific model:
    >>> chat("Hello!", model="gpt-4o-mini")  # doctest: +SKIP
    'Hello! How can I help you today?'

    With message history:
    >>> messages = [
    ...     {"role": "user", "content": "My name is Alice"},
    ...     {"role": "assistant", "content": "Nice to meet you, Alice!"},
    ...     {"role": "user", "content": "What's my name?"}
    ... ]
    >>> chat(messages, model="gpt-4o-mini")  # doctest: +SKIP
    'Your name is Alice.'

    Streaming responses:
    >>> for chunk in chat("Count to 3", stream=True):  # doctest: +SKIP
    ...     print(chunk, end='')
    1, 2, 3
"""

from collections.abc import Iterable
from typing import Union, Any

from aix.config import (
    get_config as _get_config,
    resolve_model as _resolve_model,
    ChatConfig as _ChatConfig,
)
from aix.credentials import (
    resolve_api_key as _resolve_api_key,
    requires_credentials as _requires_credentials,
)

# Import LiteLLM but keep it private - users shouldn't call it directly
try:
    from litellm import completion as _litellm_completion
    from litellm import ModelResponse as _ModelResponse
except ImportError:
    _litellm_completion = None
    _ModelResponse = None


# Shipped-default constants, kept for backward compatibility. The *active*
# defaults are resolved from ``aix.config`` at call time (see aix/config.py),
# so changing config -- not these constants -- is what affects behavior.
DFLT_CHAT_MODEL = _ChatConfig().model
DFLT_TEMPERATURE = _ChatConfig().temperature
DFLT_MAX_TOKENS = _ChatConfig().max_tokens


def _normalize_prompt(
    prompt: Union[str, Iterable[dict]],
) -> list[dict]:
    """Normalize prompt input to standard message format.

    Args:
        prompt: Either a string or list of message dicts

    Returns:
        List of message dictionaries with 'role' and 'content'

    Examples:
        >>> _normalize_prompt("Hello")
        [{'role': 'user', 'content': 'Hello'}]

        >>> _normalize_prompt([{"role": "user", "content": "Hi"}])
        [{'role': 'user', 'content': 'Hi'}]
    """
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    elif isinstance(prompt, Iterable):
        # Convert to list if it's an iterable
        messages = list(prompt)
        # Validate structure
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ValueError(
                    f"Invalid message format: {msg}. "
                    "Messages must be dicts with 'role' and 'content' keys."
                )
        return messages
    else:
        raise TypeError(
            f"Prompt must be a string or iterable of message dicts, got {type(prompt)}"
        )


def _extract_text_from_response(response: Any) -> str:
    """Extract text content from LiteLLM response.

    Args:
        response: LiteLLM ModelResponse object

    Returns:
        Extracted text content
    """
    try:
        return response.choices[0].message.content or ""
    except (AttributeError, IndexError, KeyError):
        # Fallback for unexpected response structure
        return str(response)


def _extract_text_from_stream(stream: Iterable) -> Iterable[str]:
    """Extract text chunks from LiteLLM streaming response.

    Args:
        stream: LiteLLM streaming response

    Yields:
        Text chunks as they arrive
    """
    for chunk in stream:
        try:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content is not None:
                yield delta.content
        except (AttributeError, IndexError, KeyError):
            # Skip malformed chunks
            continue


@_requires_credentials(lambda: _get_config().chat.model)
def chat(
    prompt: Union[str, Iterable[dict]],
    *,
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
    stream: bool = False,
    api_key: str = None,
    **kwargs,
) -> Union[str, Iterable[str]]:
    """Send a chat prompt and get a response.

    This is the main chat interface for AIX. It abstracts away provider-specific
    details and provides a clean, consistent API across all models.

    Args:
        prompt: Either a string (becomes a user message) or a list of message dicts
            with 'role' and 'content' keys
        model: Model identifier (e.g., 'gpt-4o', 'claude-sonnet-4',
            'openrouter/anthropic/claude-3.5-sonnet'). If None, uses default.
        temperature: Sampling temperature (0.0 = deterministic, 2.0 = creative).
            If None, uses default (1.0).
        max_tokens: Maximum tokens to generate. If None, uses model's default.
        stream: If True, return an iterator of text chunks. If False, return
            complete response as string.
        api_key: Explicit API key. If None, resolved from the environment / .env
            / AIX config store for the model's provider (see aix.credentials).
        **kwargs: Additional provider-specific parameters passed to LiteLLM

    Returns:
        If stream=False: Complete response as string
        If stream=True: Iterator yielding text chunks as they arrive

    Raises:
        ImportError: If LiteLLM is not installed
        ValueError: If prompt format is invalid

    Examples:
        >>> chat("What is Python?")  # doctest: +SKIP
        'Python is a high-level programming language...'

        >>> chat("Hello", model="gpt-4o")  # doctest: +SKIP
        'Hello! How can I assist you today?'

        >>> # Streaming response
        >>> for chunk in chat("Count to 5", stream=True):  # doctest: +SKIP
        ...     print(chunk, end='', flush=True)
        1, 2, 3, 4, 5

        >>> # With message history
        >>> messages = [
        ...     {"role": "system", "content": "You are a helpful assistant."},
        ...     {"role": "user", "content": "What is 2+2?"}
        ... ]
        >>> chat(messages)  # doctest: +SKIP
        '2+2 equals 4.'
    """
    if _litellm_completion is None:
        raise ImportError(
            "LiteLLM is required for chat functionality. "
            "Install it with: pip install litellm"
        )

    # Normalize inputs
    messages = _normalize_prompt(prompt)

    # Apply defaults from the active config (explicit args still win)
    cfg = _get_config().chat
    model = _resolve_model(model or cfg.model)
    temperature = temperature if temperature is not None else cfg.temperature
    if max_tokens is None:
        max_tokens = cfg.max_tokens

    # Build LiteLLM parameters
    litellm_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": stream,
    }

    if max_tokens is not None:
        litellm_kwargs["max_tokens"] = max_tokens

    # Resolve and inject the API key (explicit arg > env/.env > config store).
    # When None, fall back to LiteLLM's own implicit reading.
    resolved_key = _resolve_api_key(model, api_key=api_key)
    if resolved_key is not None:
        litellm_kwargs["api_key"] = resolved_key

    # Add any additional provider-specific kwargs
    litellm_kwargs.update(kwargs)

    # Call LiteLLM
    response = _litellm_completion(**litellm_kwargs)

    # Extract and return text
    if stream:
        return _extract_text_from_stream(response)
    else:
        return _extract_text_from_response(response)


def chat_with_history(
    system_prompt: str = None, *, model: str = None, **chat_kwargs
) -> "ChatSession":
    """Create a stateful chat session that maintains conversation history.

    Args:
        system_prompt: Optional system message to set context/behavior
        model: Model to use for this session
        **chat_kwargs: Additional parameters passed to chat()

    Returns:
        ChatSession object with send() method

    Examples:
        >>> session = chat_with_history("You are a helpful math tutor")  # doctest: +SKIP
        >>> session.send("What is 2+2?")  # doctest: +SKIP
        'The answer is 4.'
        >>> session.send("And if I add 3 to that?")  # doctest: +SKIP
        'That would be 7.'
        >>> len(session.history)  # doctest: +SKIP
        5
    """
    return ChatSession(system_prompt=system_prompt, model=model, **chat_kwargs)


class ChatSession:
    """Stateful chat session that maintains conversation history.

    This class provides a convenient way to have multi-turn conversations
    without manually managing message history.

    Examples:
        >>> session = ChatSession()  # doctest: +SKIP
        >>> response = session.send("My name is Alice")  # doctest: +SKIP
        >>> response = session.send("What's my name?")  # doctest: +SKIP
        'Your name is Alice.'
    """

    def __init__(self, system_prompt: str = None, *, model: str = None, **chat_kwargs):
        """Initialize a new chat session.

        Args:
            system_prompt: Optional system message
            model: Model to use for this session
            **chat_kwargs: Additional parameters for chat()
        """
        self.model = model
        self.chat_kwargs = chat_kwargs
        self.history = []

        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

    def send(self, message: str, **kwargs) -> str:
        """Send a message and get a response.

        Args:
            message: User message to send
            **kwargs: Override chat parameters for this message

        Returns:
            Assistant's response
        """
        # Add user message to history
        self.history.append({"role": "user", "content": message})

        # Merge kwargs
        call_kwargs = {**self.chat_kwargs, **kwargs}

        # Get response
        response = chat(self.history, model=self.model, **call_kwargs)

        # Add assistant response to history
        self.history.append({"role": "assistant", "content": response})

        return response

    def clear_history(self, keep_system: bool = True):
        """Clear conversation history.

        Args:
            keep_system: If True, preserve system message (if any)
        """
        if keep_system and self.history and self.history[0]["role"] == "system":
            self.history = [self.history[0]]
        else:
            self.history = []


# Convenience function for simple question-answering
def ask(question: str, model: str = None, **kwargs) -> str:
    """Ask a single question and get an answer.

    This is a convenience wrapper around chat() for simple Q&A.

    Args:
        question: The question to ask
        model: Model to use
        **kwargs: Additional parameters for chat()

    Returns:
        The answer as a string

    Examples:
        >>> from aix.chat import ask
        >>> ask("What is the capital of France?")  # doctest: +SKIP
        'The capital of France is Paris.'
    """
    return chat(question, model=model, **kwargs)
