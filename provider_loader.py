"""YAML-driven provider loader for llm_orchestrator."""

from __future__ import annotations

import inspect
import logging
import time
from typing import List, Optional

import httpx

from .config import OrchestratorConfig, ProviderConfig
from .core.base_provider import BaseProvider
from .core.provider_adapter import create_provider_adapter
from .models import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class LoadedProvider(BaseProvider):
    """Runtime provider built from config + adapter."""

    def __init__(
        self,
        provider_name: str,
        provider_config: ProviderConfig,
        supported_models: Optional[List[str]] = None,
    ):
        config_dict = {
            "api_key": provider_config.api_key,
            "base_url": provider_config.base_url,
            "timeout": provider_config.timeout,
            "default_model": provider_config.default_model,
            "supported_models": supported_models or [],
            "adapter": provider_config.adapter,
            "adapter_config": provider_config.adapter_config,
        }
        super().__init__(name=provider_name, config=config_dict)

        self._provider_name = provider_name
        self._provider_config = provider_config
        self._supported_models = supported_models or []
        self._timeout = provider_config.timeout

        self._default_model = (
            provider_config.default_model
            or (self._supported_models[0] if self._supported_models else "")
        )

        self._adapter_name = self._resolve_adapter(provider_name)
        self._adapter = None
        self._client: Optional[httpx.AsyncClient] = None
        self._init_error: Optional[str] = None
        self._validation_error: Optional[str] = None

        self._init_runtime()

    def _resolve_adapter(self, provider_name: str) -> str:
        """Resolve adapter from config or backward-compatible defaults."""
        if self._provider_config.adapter:
            return self._provider_config.adapter
        if provider_name == "anthropic":
            return "anthropic"
        if provider_name == "ollama":
            return "ollama"
        return "openai_compatible"

    def _init_runtime(self) -> None:
        try:
            self._adapter = create_provider_adapter(self._adapter_name)
            base_url = self._provider_config.base_url or self._adapter.default_base_url(
                self._provider_name
            )
            client_kwargs = {
                "timeout": self._timeout,
            }
            if base_url:
                client_kwargs["base_url"] = base_url
            self._client = httpx.AsyncClient(**client_kwargs)
        except Exception as exc:
            self._init_error = str(exc)
            logger.warning(
                "Failed to initialize provider '%s' with adapter '%s': %s",
                self._provider_name,
                self._adapter_name,
                exc,
            )

    def _resolve_request_model(self, request: LLMRequest) -> str:
        """Resolve model override from request metadata."""
        model_override = request.metadata.get("model") if request.metadata else None
        if isinstance(model_override, str) and model_override.strip():
            return model_override.strip()
        return self._default_model

    async def generate(self, request: LLMRequest) -> LLMResponse:
        start_time = time.time()
        selected_model = self._resolve_request_model(request)

        if not self.validate_config():
            self._metrics.update_error()
            return LLMResponse(
                content="",
                provider=self.name,
                model=selected_model,
                request_id=request.request_id,
                success=False,
                error_message=self._validation_error
                or self._init_error
                or f"Invalid config for provider '{self.name}'",
                latency_ms=0,
                attempted_providers=[self.name],
            )

        try:
            assert self._client is not None
            assert self._adapter is not None
            adapter_result = await self._adapter.generate(
                client=self._client,
                request=request,
                model=selected_model,
                provider_name=self.name,
                provider_config=self._provider_config,
            )
            latency_ms = (time.time() - start_time) * 1000
            self._metrics.update_success(latency_ms)
            return LLMResponse(
                content=adapter_result.content,
                provider=self.name,
                model=adapter_result.model,
                request_id=request.request_id,
                success=True,
                usage=adapter_result.usage,
                latency_ms=latency_ms,
                raw_response=adapter_result.raw_response,
                attempted_providers=[self.name],
            )
        except Exception as exc:
            latency_ms = (time.time() - start_time) * 1000
            self._metrics.update_error()
            return LLMResponse(
                content="",
                provider=self.name,
                model=selected_model,
                request_id=request.request_id,
                success=False,
                error_message=str(exc),
                latency_ms=latency_ms,
                attempted_providers=[self.name],
            )

    async def close(self) -> None:
        """Close provider clients when they expose close/aclose methods."""
        if self._client is None:
            return
        for method_name in ("aclose", "close"):
            method = getattr(self._client, method_name, None)
            if callable(method):
                result = method()
                if inspect.isawaitable(result):
                    await result
                break

    def get_available_models(self) -> list[str]:
        return list(self._supported_models)

    def validate_config(self) -> bool:
        """Validate provider runtime and adapter-specific requirements."""
        if self._init_error:
            self._validation_error = self._init_error
            return False
        if self._client is None or self._adapter is None:
            self._validation_error = "Provider runtime is not initialized"
            return False

        is_valid, error = self._adapter.validate(
            provider_name=self._provider_name,
            provider_config=self._provider_config,
            supported_models=self._supported_models,
        )
        self._validation_error = error
        return is_valid


def load_providers(config: OrchestratorConfig) -> List[BaseProvider]:
    """Load enabled providers from orchestrator config."""
    enabled_models = config.get_enabled_models()

    model_map: dict[str, list[str]] = {}
    for model in enabled_models:
        model_map.setdefault(model.provider, []).append(model.name)

    providers: List[BaseProvider] = []

    for provider_name, provider_cfg in config.providers.items():
        if not provider_cfg.enabled:
            continue

        provider = LoadedProvider(
            provider_name=provider_name,
            provider_config=provider_cfg,
            supported_models=model_map.get(provider_name, []),
        )

        if provider.validate_config():
            providers.append(provider)
        else:
            logger.warning(
                "Skipping provider '%s': invalid or incomplete configuration",
                provider_name,
            )

    return providers
