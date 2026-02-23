from __future__ import annotations

from typing import Any

from llm_orchestrator.core import BaseProvider
from llm_orchestrator.models import LLMRequest, LLMResponse


class FakeProvider(BaseProvider):
    def __init__(
        self,
        name: str,
        *,
        healthy: bool = True,
        fail_times: int = 0,
        latency_ms: float = 10.0,
    ) -> None:
        super().__init__(name=name, config={})
        self._healthy = healthy
        self._fail_times = fail_times
        self._latency_ms = latency_ms
        self.calls = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        if self.calls <= self._fail_times:
            self._metrics.update_error()
            return LLMResponse(
                content="",
                provider=self.name,
                model="fake-model",
                request_id=request.request_id,
                success=False,
                error_message=f"{self.name} failed",
                latency_ms=self._latency_ms,
                attempted_providers=[self.name],
            )

        self._metrics.update_success(self._latency_ms)
        return LLMResponse(
            content=f"{self.name}:{request.prompt}",
            provider=self.name,
            model="fake-model",
            request_id=request.request_id,
            success=True,
            latency_ms=self._latency_ms,
            attempted_providers=[self.name],
            usage={"total_tokens": 1},
        )

    def get_available_models(self) -> list[str]:
        return ["fake-model"]

    def validate_config(self) -> bool:
        return self._healthy


def set_provider_latency(provider: BaseProvider, latency_ms: float) -> None:
    metrics = provider.get_metrics()
    metrics.avg_latency_ms = latency_ms


def set_provider_success_rate(provider: BaseProvider, success_rate: float) -> None:
    metrics = provider.get_metrics()
    metrics.success_rate = success_rate
    metrics.total_requests = 100
    metrics.error_count = int((1.0 - success_rate) * 100)