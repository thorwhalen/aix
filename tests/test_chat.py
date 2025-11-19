"""Tests for aix.chat module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from aix.chat import (
    chat,
    ask,
    chat_with_history,
    ChatSession,
    _normalize_prompt,
    _extract_text_from_response,
)


class TestNormalizePrompt:
    """Tests for prompt normalization."""

    def test_string_prompt(self):
        """Test normalizing a string prompt."""
        result = _normalize_prompt("Hello")
        assert result == [{"role": "user", "content": "Hello"}]

    def test_message_list(self):
        """Test normalizing a message list."""
        messages = [{"role": "user", "content": "Hi"}]
        result = _normalize_prompt(messages)
        assert result == messages

    def test_invalid_message_format(self):
        """Test that invalid message format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid message format"):
            _normalize_prompt([{"invalid": "message"}])

    def test_invalid_type(self):
        """Test that invalid type raises TypeError."""
        with pytest.raises(TypeError):
            _normalize_prompt(123)


class TestChat:
    """Tests for chat function."""

    @patch('aix.chat._litellm_completion')
    def test_simple_chat(self, mock_completion):
        """Test simple chat with string prompt."""
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Test response"
        mock_completion.return_value = mock_response

        result = chat("Hello")

        assert result == "Test response"
        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs['messages'] == [{"role": "user", "content": "Hello"}]

    @patch('aix.chat._litellm_completion')
    def test_chat_with_model(self, mock_completion):
        """Test chat with specific model."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Response"
        mock_completion.return_value = mock_response

        chat("Hello", model="gpt-4o")

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs['model'] == "gpt-4o"

    @patch('aix.chat._litellm_completion')
    def test_chat_with_temperature(self, mock_completion):
        """Test chat with custom temperature."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Response"
        mock_completion.return_value = mock_response

        chat("Hello", temperature=0.5)

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs['temperature'] == 0.5

    @patch('aix.chat._litellm_completion')
    def test_chat_streaming(self, mock_completion):
        """Test streaming chat."""
        # Mock streaming response
        mock_chunk1 = Mock()
        mock_chunk1.choices = [Mock()]
        mock_chunk1.choices[0].delta = Mock()
        mock_chunk1.choices[0].delta.content = "Hello"

        mock_chunk2 = Mock()
        mock_chunk2.choices = [Mock()]
        mock_chunk2.choices[0].delta = Mock()
        mock_chunk2.choices[0].delta.content = " world"

        mock_completion.return_value = [mock_chunk1, mock_chunk2]

        result = list(chat("Test", stream=True))

        assert result == ["Hello", " world"]
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs['stream'] is True

    @patch('aix.chat._litellm_completion')
    def test_chat_with_message_history(self, mock_completion):
        """Test chat with message history."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Response"
        mock_completion.return_value = mock_response

        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"}
        ]

        chat(messages)

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs['messages'] == messages


class TestAsk:
    """Tests for ask convenience function."""

    @patch('aix.chat.chat')
    def test_ask(self, mock_chat):
        """Test ask function."""
        mock_chat.return_value = "Paris"

        result = ask("What is the capital of France?")

        assert result == "Paris"
        mock_chat.assert_called_once_with(
            "What is the capital of France?",
            model=None
        )


class TestChatSession:
    """Tests for ChatSession class."""

    @patch('aix.chat.chat')
    def test_session_initialization(self, mock_chat):
        """Test session initialization."""
        session = ChatSession()
        assert session.history == []

    @patch('aix.chat.chat')
    def test_session_with_system_prompt(self, mock_chat):
        """Test session with system prompt."""
        session = ChatSession(system_prompt="You are helpful")
        assert len(session.history) == 1
        assert session.history[0]['role'] == 'system'

    @patch('aix.chat.chat')
    def test_session_send(self, mock_chat):
        """Test sending message in session."""
        mock_chat.return_value = "I'm doing well"

        session = ChatSession()
        response = session.send("How are you?")

        assert response == "I'm doing well"
        assert len(session.history) == 2
        assert session.history[0]['role'] == 'user'
        assert session.history[1]['role'] == 'assistant'

    @patch('aix.chat.chat')
    def test_session_maintains_history(self, mock_chat):
        """Test that session maintains history across multiple sends."""
        mock_chat.side_effect = ["Hello!", "Your name is Alice"]

        session = ChatSession()
        session.send("Hi")
        session.send("What's my name?")

        # Should have 4 messages: 2 user, 2 assistant
        assert len(session.history) == 4

        # Verify chat was called with full history
        last_call = mock_chat.call_args[0][0]
        assert len(last_call) == 3  # Including the new user message

    @patch('aix.chat.chat')
    def test_session_clear_history(self, mock_chat):
        """Test clearing session history."""
        mock_chat.return_value = "Response"

        session = ChatSession(system_prompt="You are helpful")
        session.send("Hello")

        session.clear_history(keep_system=True)
        assert len(session.history) == 1
        assert session.history[0]['role'] == 'system'

        session.clear_history(keep_system=False)
        assert len(session.history) == 0


class TestChatWithHistory:
    """Tests for chat_with_history function."""

    def test_creates_session(self):
        """Test that chat_with_history creates a ChatSession."""
        session = chat_with_history("Be helpful")
        assert isinstance(session, ChatSession)
        assert len(session.history) == 1
        assert session.history[0]['content'] == "Be helpful"
