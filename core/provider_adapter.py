"""Provider adapter abstractions for vendor-neutral LLM integration."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

import httpx

from ..config import ProviderConfig
from ..models import LLMRequest


@dataclass
class AdapterResult:
    """Normalized adapter output consumed by the orchestrator."""

    content: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    raw_response: Optional[Dict[str, Any]] = None


class ProviderAdapter(ABC):
    """Abstract provider adapter contract."""

    @abstractmethod
    async def generate(
        self,
        *,
        client: httpx.AsyncClient,
        request: LLMRequest,
        model: str,
        provider_name: str,
        provider_config: ProviderConfig,
    ) -> AdapterResult:
        """Run request against a provider and return normalized output."""

    def default_base_url(self, provider_name: str) -> Optional[str]:
        """Optional default base URL for an adapter/provider."""
        _ = provider_name
        return None

    def validate(
        self,
        *,
        provider_name: str,
        provider_config: ProviderConfig,
        supported_models: list[str],
    ) -> tuple[bool, Optional[str]]:
        """Validate configuration before provider is considered healthy."""
        _ = provider_name
        _ = supported_models
        if not provider_config.default_model:
            return False, "Missing default_model"
        return True, None


def _extract_text_content(value: Any) -> str:
    """Best-effort content normalization from various JSON payloads."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    if isinstance(value, dict):
        for key in ("text", "content", "response", "output_text"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
    return str(value)


class OpenAICompatibleAdapter(ProviderAdapter):
    """Adapter for OpenAI-compatible chat completion APIs."""

    def default_base_url(self, provider_name: str) -> Optional[str]:
        if provider_name == "openai":
            return "https://api.openai.com/v1"
        return None

    def validate(
        self,
        *,
        provider_name: str,
        provider_config: ProviderConfig,
        supported_models: list[str],
    ) -> tuple[bool, Optional[str]]:
        _ = supported_models
        if not provider_config.api_key:
            return False, f"Provider '{provider_name}' requires api_key"
        if not provider_config.default_model:
            return False, f"Provider '{provider_name}' requires default_model"
        if not provider_config.base_url and provider_name != "openai":
            return False, f"Provider '{provider_name}' requires base_url"
        return True, None

    async def generate(
        self,
        *,
        client: httpx.AsyncClient,
        request: LLMRequest,
        model: str,
        provider_name: str,
        provider_config: ProviderConfig,
    ) -> AdapterResult:
        _ = provider_name
        adapter_cfg = provider_config.adapter_config or {}
        endpoint = str(adapter_cfg.get("chat_completions_path", "/chat/completions"))

        messages: list[dict[str, str]] = [{"role": "user", "content": request.prompt}]
        if request.system_prompt:
            messages.insert(0, {"role": "system", "content": request.system_prompt})

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        payload.update(adapter_cfg.get("extra_body", {}) or {})

        headers: Dict[str, str] = dict(adapter_cfg.get("headers", {}) or {})
        if provider_config.api_key:
            headers["Authorization"] = f"Bearer {provider_config.api_key}"

        response = await client.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        content = ""
        choices = data.get("choices", [])
        if choices:
            first = choices[0] or {}
            message = first.get("message", {})
            content = _extract_text_content(message.get("content"))

        usage = None
        usage_data = data.get("usage")
        if isinstance(usage_data, dict):
            usage = {
                "input_tokens": int(usage_data.get("prompt_tokens", 0) or 0),
                "output_tokens": int(usage_data.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage_data.get("total_tokens", 0) or 0),
            }

        return AdapterResult(
            content=content,
            model=str(data.get("model", model)),
            usage=usage,
            raw_response=data if isinstance(data, dict) else {"response": data},
        )


class AnthropicAdapter(ProviderAdapter):
    """Adapter for Anthropic Messages HTTP API."""

    def default_base_url(self, provider_name: str) -> Optional[str]:
        _ = provider_name
        return "https://api.anthropic.com"

    def validate(
        self,
        *,
        provider_name: str,
        provider_config: ProviderConfig,
        supported_models: list[str],
    ) -> tuple[bool, Optional[str]]:
        _ = supported_models
        if not provider_config.api_key:
            return False, f"Provider '{provider_name}' requires api_key"
        if not provider_config.default_model:
            return False, f"Provider '{provider_name}' requires default_model"
        return True, None

    async def generate(
        self,
        *,
        client: httpx.AsyncClient,
        request: LLMRequest,
        model: str,
        provider_name: str,
        provider_config: ProviderConfig,
    ) -> AdapterResult:
        _ = provider_name
        adapter_cfg = provider_config.adapter_config or {}
        endpoint = str(adapter_cfg.get("messages_path", "/v1/messages"))
        anthropic_version = str(adapter_cfg.get("anthropic_version", "2023-06-01"))

        payload: Dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.system_prompt:
            payload["system"] = request.system_prompt
        payload.update(adapter_cfg.get("extra_body", {}) or {})

        headers: Dict[str, str] = dict(adapter_cfg.get("headers", {}) or {})
        headers["x-api-key"] = str(provider_config.api_key)
        headers["anthropic-version"] = anthropic_version

        response = await client.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        content = _extract_text_content(data.get("content"))
        usage = None
        usage_data = data.get("usage")
        if isinstance(usage_data, dict):
            input_tokens = int(usage_data.get("input_tokens", 0) or 0)
            output_tokens = int(usage_data.get("output_tokens", 0) or 0)
            usage = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }

        return AdapterResult(
            content=content,
            model=str(data.get("model", model)),
            usage=usage,
            raw_response=data if isinstance(data, dict) else {"response": data},
        )


class OllamaAdapter(ProviderAdapter):
    """Adapter for local/remote Ollama HTTP API."""

    def default_base_url(self, provider_name: str) -> Optional[str]:
        _ = provider_name
        return "http://localhost:11434"

    async def generate(
        self,
        *,
        client: httpx.AsyncClient,
        request: LLMRequest,
        model: str,
        provider_name: str,
        provider_config: ProviderConfig,
    ) -> AdapterResult:
        _ = provider_name
        adapter_cfg = provider_config.adapter_config or {}
        endpoint = str(adapter_cfg.get("generate_path", "/api/generate"))

        payload: Dict[str, Any] = {
            "model": model,
            "prompt": request.prompt,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }
        if request.system_prompt:
            payload["system"] = request.system_prompt
        payload.update(adapter_cfg.get("extra_body", {}) or {})

        headers: Dict[str, str] = dict(adapter_cfg.get("headers", {}) or {})
        if provider_config.api_key:
            headers["Authorization"] = f"Bearer {provider_config.api_key}"

        response = await client.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        input_tokens = int(data.get("prompt_eval_count", 0) or 0)
        output_tokens = int(data.get("eval_count", 0) or 0)
        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

        return AdapterResult(
            content=_extract_text_content(data.get("response", "")),
            model=str(data.get("model", model)),
            usage=usage,
            raw_response=data if isinstance(data, dict) else {"response": data},
        )


_ADAPTER_REGISTRY: Dict[str, Type[ProviderAdapter]] = {
    "openai_compatible": OpenAICompatibleAdapter,
    "anthropic": AnthropicAdapter,
    "ollama": OllamaAdapter,
}


def register_provider_adapter(name: str, adapter_cls: Type[ProviderAdapter]) -> None:
    """Register a custom provider adapter class globally."""
    _ADAPTER_REGISTRY[name] = adapter_cls


def _load_class_from_path(path: str) -> type[Any]:
    """Load class from 'module:ClassName' or 'module.ClassName' path."""
    if ":" in path:
        module_name, class_name = path.split(":", 1)
    else:
        module_name, _, class_name = path.rpartition(".")
        if not module_name:
            raise ValueError(f"Invalid class path: {path}")
    module = importlib.import_module(module_name)
    loaded = getattr(module, class_name, None)
    if loaded is None:
        raise ValueError(f"Class '{class_name}' not found in module '{module_name}'")
    if not isinstance(loaded, type):
        raise ValueError(f"Loaded object is not a class: {path}")
    return loaded


def create_provider_adapter(adapter_name_or_path: str) -> ProviderAdapter:
    """Create adapter instance from registry name or class import path."""
    adapter_cls = _ADAPTER_REGISTRY.get(adapter_name_or_path)
    if adapter_cls is None:
        loaded_cls = _load_class_from_path(adapter_name_or_path)
        if not issubclass(loaded_cls, ProviderAdapter):
            raise ValueError(
                f"Adapter '{adapter_name_or_path}' must subclass ProviderAdapter"
            )
        adapter_cls = loaded_cls
    return adapter_cls()
