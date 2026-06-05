"""Tests for unified credential resolution (aix.credentials) and its wiring.

Covers the resolution layers, provider inference, the actionable
``MissingCredentialError``, the ``requires_credentials`` preflight decorator,
the ``check_keys`` doctor, and end-to-end ``api_key=`` injection into the
LiteLLM call from ``chat``.
"""

import sys
from unittest.mock import Mock, patch

import pytest

from aix.credentials import (
    PROVIDER_ENV_VARS,
    MissingCredentialError,
    check_keys,
    check_requirements,
    infer_provider,
    provider_env_vars,
    requires_credentials,
    resolve_api_key,
)

_aix_chat = sys.modules["aix.chat"]


@pytest.fixture
def no_keys(monkeypatch):
    """Clear every known provider env var and make the config store look empty."""
    for provider in PROVIDER_ENV_VARS:
        for env_name in provider_env_vars(provider):
            monkeypatch.delenv(env_name, raising=False)
    # Don't let a real local config store (or a project .env) leak a key in.
    monkeypatch.setattr("aix.credentials._lookup_store", lambda name: None)
    monkeypatch.setattr("aix.credentials._dotenv_loaded", True)


class TestProviderInference:
    def test_infer_from_model(self):
        assert infer_provider("gpt-4o") == "openai"
        assert infer_provider("claude-sonnet-4-20250514") == "anthropic"

    def test_provider_name_passthrough(self):
        assert infer_provider("openrouter") == "openrouter"

    def test_none(self):
        assert infer_provider(None) is None

    def test_env_var_aliases(self):
        # gemini accepts both GEMINI_ and GOOGLE_ names.
        assert provider_env_vars("gemini") == ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
        assert provider_env_vars("totally-unknown") == []


class TestResolveApiKey:
    def test_explicit_arg_wins(self, no_keys):
        assert resolve_api_key("gpt-4o", api_key="sk-explicit") == "sk-explicit"

    def test_env_var(self, no_keys, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        assert resolve_api_key("gpt-4o") == "sk-from-env"

    def test_store_fallback(self, no_keys, monkeypatch):
        monkeypatch.setattr(
            "aix.credentials._lookup_store",
            lambda name: "sk-from-store" if name == "OPENAI_API_KEY" else None,
        )
        assert resolve_api_key("gpt-4o") == "sk-from-store"

    def test_env_beats_store(self, no_keys, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        monkeypatch.setattr("aix.credentials._lookup_store", lambda name: "sk-store")
        assert resolve_api_key("gpt-4o") == "sk-from-env"

    def test_missing_returns_none(self, no_keys):
        assert resolve_api_key("gpt-4o") is None


class TestCheckRequirements:
    def test_passes_with_key(self, no_keys, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        assert check_requirements("gpt-4o") is True

    def test_raises_when_missing(self, no_keys):
        with pytest.raises(MissingCredentialError) as exc:
            check_requirements("gpt-4o")
        msg = str(exc.value)
        # Names which key, how to set it, and where to get one.
        assert "OPENAI_API_KEY" in msg
        assert "export" in msg
        assert "platform.openai.com" in msg

    def test_error_never_leaks_value(self, no_keys):
        err = MissingCredentialError("gpt-4o")
        assert "sk-" not in str(err)


class TestRequiresCredentialsDecorator:
    def test_runs_when_key_present(self, no_keys, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")

        @requires_credentials(lambda: "gpt-4o")
        def f(*, model=None, api_key=None):
            return "ran"

        assert f() == "ran"

    def test_raises_when_missing(self, no_keys):
        @requires_credentials(lambda: "gpt-4o")
        def f(*, model=None, api_key=None):
            return "ran"

        with pytest.raises(MissingCredentialError):
            f()

    def test_explicit_api_key_satisfies_preflight(self, no_keys):
        @requires_credentials(lambda: "gpt-4o")
        def f(*, model=None, api_key=None):
            return "ran"

        assert f(api_key="sk-explicit") == "ran"


class TestCheckKeys:
    def test_reports_availability_without_values(self, no_keys, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
        report = check_keys(["openai", "anthropic"])
        assert report["openai"]["available"] is True
        assert report["openai"]["source"] == "env"
        assert report["anthropic"]["available"] is False
        # No key value anywhere in the report.
        assert "sk-secret" not in repr(report)


class TestChatKeyInjection:
    """End-to-end: chat injects the resolved api_key into the LiteLLM call."""

    @patch.object(_aix_chat, "_litellm_completion")
    def test_explicit_api_key_injected(self, mock_completion, no_keys):
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "ok"
        mock_completion.return_value = mock_response

        _aix_chat.chat("hi", model="gpt-4o", api_key="sk-explicit")

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["api_key"] == "sk-explicit"

    @patch.object(_aix_chat, "_litellm_completion")
    def test_env_api_key_injected(self, mock_completion, no_keys, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "ok"
        mock_completion.return_value = mock_response

        _aix_chat.chat("hi", model="gpt-4o")

        assert mock_completion.call_args[1]["api_key"] == "sk-from-env"

    @patch.object(_aix_chat, "_litellm_completion")
    def test_missing_key_raises_before_litellm(self, mock_completion, no_keys):
        with pytest.raises(MissingCredentialError):
            _aix_chat.chat("hi", model="gpt-4o")
        mock_completion.assert_not_called()
