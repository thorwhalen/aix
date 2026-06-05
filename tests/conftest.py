"""Pytest configuration and shared fixtures."""

import sys
import pytest
from unittest.mock import Mock, patch

# `aix/__init__.py` exports `chat`/`embeddings` (the function) under the same
# name as their submodule, so `aix.chat` resolves to the function on some
# Python builds. Patch the submodule objects directly to stay version-robust.
import aix.chat  # noqa: F401
import aix.embeddings  # noqa: F401

_aix_chat = sys.modules["aix.chat"]
_aix_embeddings = sys.modules["aix.embeddings"]


@pytest.fixture(autouse=True)
def dummy_provider_keys(monkeypatch):
    """Provide dummy provider API keys for the whole suite.

    Credential preflight (``aix.credentials.requires_credentials``) now raises
    ``MissingCredentialError`` when a key is genuinely absent. Existing tests mock
    the LiteLLM call but run without real keys, so we inject placeholders here;
    LiteLLM is mocked, so the dummy values are never used over the wire.

    Credential-specific tests opt out by clearing the relevant env vars
    (``monkeypatch.delenv``) and patching the config-store lookup. Video
    providers (runway/pika) are deliberately excluded: test_video asserts on
    their presence/absence, so the suite must not seed them globally.
    """
    from aix.credentials import PROVIDER_ENV_VARS, provider_env_vars

    excluded = {"runway", "pika"}
    for provider in PROVIDER_ENV_VARS:
        if provider in excluded:
            continue
        for env_name in provider_env_vars(provider):
            monkeypatch.setenv(env_name, "sk-test-dummy-key")


@pytest.fixture
def mock_litellm_completion():
    """Mock LiteLLM completion function."""
    with patch.object(_aix_chat, "_litellm_completion") as mock:
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Test response"
        mock.return_value = mock_response
        yield mock


@pytest.fixture
def mock_litellm_embedding():
    """Mock LiteLLM embedding function."""
    with patch.object(_aix_embeddings, "_litellm_embedding") as mock:
        mock_response = Mock()
        mock_response.data = [{"embedding": [0.1, 0.2, 0.3]}]
        mock.return_value = mock_response
        yield mock


@pytest.fixture
def sample_models():
    """Sample model objects for testing."""
    from aix.ai_models import Model

    return [
        Model(
            id="openai/gpt-4",
            provider="openai",
            context_size=8192,
            cost_per_token={"input": 0.00001, "output": 0.00003},
            is_local=False,
        ),
        Model(
            id="openai/gpt-3.5-turbo",
            provider="openai",
            context_size=4096,
            cost_per_token={"input": 0.000001, "output": 0.000002},
            is_local=False,
        ),
        Model(
            id="anthropic/claude-3",
            provider="anthropic",
            context_size=200000,
            cost_per_token={"input": 0.00002, "output": 0.00006},
            is_local=False,
        ),
    ]


@pytest.fixture
def mock_model_manager(sample_models):
    """Mock ModelManager with sample models."""
    from aix.ai_models import ModelRegistry

    manager = Mock()
    registry = ModelRegistry()

    for model in sample_models:
        registry[model.id] = model

    manager.models = registry
    manager.discover_from_source = Mock(return_value=sample_models)
    manager.list_models = Mock(return_value=sample_models)
    manager.get_connector_metadata = Mock(return_value={"model": "test"})

    return manager
