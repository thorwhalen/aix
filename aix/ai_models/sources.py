"""Concrete implementations of model sources and connectors.

This module provides ready-to-use sources for discovering models
and connectors for formatting metadata for different clients.
"""

import requests
from typing import Any, Iterable
from dataclasses import dataclass

from aix.ai_models.base import Model, ModelSource, Connector


class OpenRouterSource(ModelSource):
    """Discover models from OpenRouter's public API.
    
    OpenRouter maintains a comprehensive registry of models across providers.
    """
    
    def __init__(self, *, timeout: int = 30):
        self._timeout = timeout
        self._endpoint = "https://openrouter.ai/api/v1/models"
    
    def discover_models(self) -> Iterable[Model]:
        """Fetch models from OpenRouter API.
        
        >>> source = OpenRouterSource()
        >>> models = list(source.discover_models())
        >>> len(models) > 0  # Should find many models
        True
        """
        response = requests.get(self._endpoint, timeout=self._timeout)
        response.raise_for_status()
        data = response.json()
        
        for item in data.get("data", []):
            model_id = item.get("id") or item.get("name")
            if not model_id:
                continue
            
            # Extract provider info
            provider_data = item.get("model", {})
            provider = provider_data.get("provider", "")
            
            # Extract context size
            context_size = item.get("context_length") or provider_data.get("context_length")
            
            # Extract pricing
            pricing = item.get("pricing", {})
            cost_per_token = {}
            if "prompt" in pricing:
                cost_per_token["input"] = float(pricing["prompt"])
            if "completion" in pricing:
                cost_per_token["output"] = float(pricing["completion"])
            
            # Build capabilities dict
            capabilities = {}
            if item.get("supports_streaming"):
                capabilities["streaming"] = True
            if item.get("supports_functions"):
                capabilities["function_calling"] = True
            if item.get("supports_vision"):
                capabilities["vision"] = True
            
            yield Model(
                id=model_id,
                provider=provider,
                context_size=context_size,
                is_local=False,
                capabilities=capabilities,
                cost_per_token=cost_per_token,
                connector_metadata={
                    "openrouter": {"id": model_id, "name": item.get("name")}
                }
            )


class OllamaSource(ModelSource):
    """Discover locally installed Ollama models.
    
    Queries the Ollama API for models available on the local machine.
    """
    
    def __init__(self, *, base_url: str = "http://localhost:11434", timeout: int = 10):
        self._base_url = base_url
        self._timeout = timeout
    
    def discover_models(self) -> Iterable[Model]:
        """Fetch locally installed Ollama models.
        
        >>> source = OllamaSource()
        >>> models = list(source.discover_models())  # May be empty if Ollama not running
        >>> all(m.is_local for m in models)
        True
        """
        try:
            response = requests.get(
                f"{self._base_url}/api/tags",
                timeout=self._timeout
            )
            response.raise_for_status()
            data = response.json()
            
            for item in data.get("models", []):
                model_name = item.get("name")
                if not model_name:
                    continue
                
                # Ollama models are always local
                yield Model(
                    id=f"ollama/{model_name}",
                    provider="ollama",
                    is_local=True,
                    connector_metadata={
                        "ollama": {
                            "name": model_name,
                            "base_url": self._base_url
                        }
                    },
                    custom_metadata={
                        "size": item.get("size"),
                        "modified_at": item.get("modified_at")
                    }
                )
        except (requests.RequestException, ValueError):
            # Ollama might not be running
            return


class ProviderAPISource(ModelSource):
    """Discover models from a provider's API (OpenAI, Anthropic, etc.).
    
    Generic source for providers with /v1/models endpoint.
    """
    
    def __init__(
        self,
        provider_name: str,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 30
    ):
        self._provider_name = provider_name
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
    
    def discover_models(self) -> Iterable[Model]:
        """Fetch models from provider API.
        
        Requires valid API key for the provider.
        """
        headers = {"Authorization": f"Bearer {self._api_key}"}
        
        try:
            response = requests.get(
                f"{self._base_url}/models",
                headers=headers,
                timeout=self._timeout
            )
            response.raise_for_status()
            data = response.json()
            
            for item in data.get("data", []):
                model_id = item.get("id")
                if not model_id:
                    continue
                
                yield Model(
                    id=model_id,
                    provider=self._provider_name,
                    is_local=False,
                    connector_metadata={
                        self._provider_name: {
                            "id": model_id,
                            "base_url": self._base_url
                        }
                    },
                    custom_metadata={
                        "owned_by": item.get("owned_by"),
                        "created": item.get("created")
                    }
                )
        except (requests.RequestException, ValueError):
            return


# === Connectors ===


class OpenAIConnector(Connector):
    """Format metadata for OpenAI Python client.
    
    Provides model metadata in format expected by openai.ChatCompletion.create().
    """
    
    @property
    def name(self) -> str:
        return "openai"
    
    def format_metadata(self, model: "Model") -> dict[str, Any]:
        return {"model": model.id}


class OpenRouterConnector(Connector):
    """Connector for OpenRouter API."""

    @property
    def name(self) -> str:
        return "openrouter"

    def format_metadata(self, model: "Model") -> dict[str, Any]:
        """Formats metadata for OpenRouter, which uses the model ID directly."""
        return {"model": model.id}


class LangChainConnector(Connector):
    """Connector for LangChain library."""
    
    @property
    def name(self) -> str:
        return "langchain"
    
    def format_metadata(self, model: Model) -> dict[str, Any]:
        """Format for LangChain.
        
        >>> connector = LangChainConnector()
        >>> model = Model(id="gpt-4", provider="openai")
        >>> metadata = connector.format_metadata(model)
        >>> metadata["model_name"]
        'gpt-4'
        """
        result = {
            "model_name": model.id,
        }
        
        # Add provider-specific class hint
        if model.provider == "openai":
            result["_class"] = "ChatOpenAI"
        elif model.provider == "anthropic":
            result["_class"] = "ChatAnthropic"
        elif model.provider == "ollama":
            result["_class"] = "ChatOllama"
            ollama_meta = model.connector_metadata.get("ollama", {})
            result["base_url"] = ollama_meta.get("base_url", "http://localhost:11434")
            result["model"] = ollama_meta.get("name", model.id)
        
        if model.context_size:
            result["max_tokens"] = model.context_size
        
        return result


class OllamaConnector(Connector):
    """Format metadata for direct Ollama API calls."""
    
    @property
    def name(self) -> str:
        return "ollama"
    
    def format_metadata(self, model: Model) -> dict[str, Any]:
        """Format for Ollama API.
        
        >>> connector = OllamaConnector()
        >>> model = Model(id="ollama/llama2", provider="ollama", is_local=True)
        >>> model.connector_metadata["ollama"] = {"name": "llama2"}
        >>> metadata = connector.format_metadata(model)
        >>> metadata["model"]
        'llama2'
        """
        ollama_meta = model.connector_metadata.get("ollama", {})
        
        result = {
            "model": ollama_meta.get("name", model.id.replace("ollama/", "")),
            "base_url": ollama_meta.get("base_url", "http://localhost:11434"),
        }
        
        return result


class DSPyConnector(Connector):
    """Format metadata for DSPy framework."""
    
    @property
    def name(self) -> str:
        return "dspy"
    
    def format_metadata(self, model: Model) -> dict[str, Any]:
        """Format for DSPy.
        
        >>> connector = DSPyConnector()
        >>> model = Model(id="gpt-4", provider="openai")
        >>> metadata = connector.format_metadata(model)
        >>> "model" in metadata
        True
        """
        result = {
            "model": model.id,
        }
        
        # DSPy uses different classes for different providers
        if model.provider == "openai":
            result["_class"] = "dspy.OpenAI"
        elif model.provider == "anthropic":
            result["_class"] = "dspy.Claude"
        elif model.provider == "ollama":
            result["_class"] = "dspy.Ollama"
            ollama_meta = model.connector_metadata.get("ollama", {})
            result["model"] = ollama_meta.get("name", model.id)
        
        if model.context_size:
            result["max_tokens"] = model.context_size
        
        return result
