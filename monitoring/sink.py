"""Observability sink interface for custom telemetry backends."""

from __future__ import annotations

from typing import Any, Optional


class ObservabilitySink:
    """Base sink with no-op event handlers."""

    def on_request_start(
        self,
        *,
        request_id: str,
        routing_strategy: str,
        task_complexity: str = "",
        provider_hint: str = "",
        model_hint: str = "",
    ) -> Optional[Any]:
        _ = request_id
        _ = routing_strategy
        _ = task_complexity
        _ = provider_hint
        _ = model_hint
        return None

    def on_cache_lookup(self, *, hit: bool) -> None:
        _ = hit

    def on_provider_attempt(
        self,
        *,
        provider: str,
        success: bool,
        latency_ms: float,
        model: str = "",
        error_message: Optional[str] = None,
        request_span: Optional[Any] = None,
    ) -> None:
        _ = provider
        _ = success
        _ = latency_ms
        _ = model
        _ = error_message
        _ = request_span

    def on_request_end(
        self,
        *,
        provider: str,
        model: str,
        success: bool,
        latency_ms: float,
        cached: bool,
        fallback_used: bool,
        error_message: Optional[str] = None,
        request_span: Optional[Any] = None,
    ) -> None:
        _ = provider
        _ = model
        _ = success
        _ = latency_ms
        _ = cached
        _ = fallback_used
        _ = error_message
        _ = request_span

    def get_status(self) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        return None
