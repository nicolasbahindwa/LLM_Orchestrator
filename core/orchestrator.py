"""Main orchestrator engine."""

from __future__ import annotations

import inspect
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..cache import LLMCache
from ..config import ComplexityLevel
from ..monitoring import RequestMetrics
from ..models import LLMRequest, LLMResponse
from ..routing import Router, RoutingStrategy
from .base_provider import BaseProvider


class Orchestrator:
    """Coordinates routing, cache, provider execution, and fallback."""

    def __init__(
        self,
        providers: List[BaseProvider],
        cache: Optional[LLMCache] = None,
        router: Optional[Router] = None,
        max_fallback_attempts: int = 3,
    ):
        self._providers = providers
        self._cache = cache
        self._max_fallback_attempts = max_fallback_attempts
        self._router = router or Router(providers, strategy=RoutingStrategy.BALANCED)

        # Optional components set by factory
        self.retry_handler = None
        self.logger = None
        self.complexity_analyzer = None
        self.model_router = None
        self.metrics_collector = None
        self.observability = None
        self.circuit_breakers: Dict[str, Any] = {}

        # Runtime stats
        self._total_requests = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._fallback_uses = 0

    def _log(self, level: str, message: str, **extra: Any) -> None:
        if not self.logger:
            return
        method = getattr(self.logger, level, None)
        if callable(method):
            method(message, **extra)

    async def _execute_provider(self, provider: BaseProvider, request: LLMRequest) -> LLMResponse:
        async def run_call() -> LLMResponse:
            if self.retry_handler:
                return await self.retry_handler.execute(provider.generate, request)
            return await provider.generate(request)

        try:
            breaker = self.circuit_breakers.get(provider.name)
            if breaker:
                return await breaker.execute(run_call)
            return await run_call()
        except Exception as exc:
            provider.get_metrics().update_error()
            return LLMResponse(
                content="",
                provider=provider.name,
                model="",
                request_id=request.request_id,
                success=False,
                error_message=str(exc),
                attempted_providers=[provider.name],
            )

    def _apply_model_routing(self, request: LLMRequest, complexity: Optional[Any]) -> LLMRequest:
        """Use model router output to hint provider/model selection for this request."""
        if not self.model_router or not complexity:
            return request

        try:
            task_complexity = ComplexityLevel(complexity.value)
            selected_model = self.model_router.select_model(task_complexity=task_complexity)
            if not selected_model:
                return request

            cloned = LLMRequest(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system_prompt=request.system_prompt,
                priority=request.priority,
                metadata=dict(request.metadata),
                request_id=request.request_id,
            )
            cloned.metadata["provider"] = selected_model.provider
            cloned.metadata["model"] = selected_model.name
            return cloned
        except Exception:
            return request

    def _record_request_metrics(
        self,
        request: LLMRequest,
        response: LLMResponse,
        start_time: float,
        complexity: Optional[Any],
        routing_strategy: Optional[RoutingStrategy],
    ) -> None:
        if not self.metrics_collector:
            return

        usage = response.usage or {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)

        strategy_name = self._resolve_strategy_name(routing_strategy)

        metrics = RequestMetrics(
            request_id=request.request_id,
            provider=response.provider,
            model=response.model,
            start_time=datetime.fromtimestamp(start_time),
            end_time=datetime.now(),
            latency_ms=response.latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            success=response.success,
            error=response.error_message,
            cached=response.cached,
            fallback_used=response.fallback_used,
            routing_strategy=strategy_name,
            task_complexity=complexity.value if complexity else "",
        )
        self.metrics_collector.record_request(metrics)

    def _resolve_strategy_name(self, routing_strategy: Optional[RoutingStrategy]) -> str:
        """Resolve routing strategy name from explicit or router default strategy."""
        if routing_strategy:
            return routing_strategy.value
        if hasattr(self._router, "_strategy"):
            strategy = getattr(self._router, "_strategy")
            return getattr(strategy, "value", str(strategy))
        return ""

    async def generate(
        self,
        request: LLMRequest,
        routing_strategy: Optional[RoutingStrategy] = None,
    ) -> LLMResponse:
        self._total_requests += 1
        start_time = time.time()
        request_span = None

        complexity = None
        if self.complexity_analyzer:
            try:
                complexity = self.complexity_analyzer.analyze(request.prompt)
            except Exception:
                complexity = None

        self._log(
            "debug",
            "Request received",
            request_id=request.request_id,
            complexity=complexity.value if complexity else "unknown",
        )
        request = self._apply_model_routing(request, complexity)

        if self.observability:
            request_span = self.observability.on_request_start(
                request_id=request.request_id,
                routing_strategy=self._resolve_strategy_name(routing_strategy),
                task_complexity=complexity.value if complexity else "",
                provider_hint=str(request.metadata.get("provider", "")),
                model_hint=str(request.metadata.get("model", "")),
            )

        # Cache lookup
        if self._cache:
            cached = self._cache.get(request)
            if cached:
                if self.observability:
                    self.observability.on_cache_lookup(hit=True)
                self._cache_hits += 1
                self._successful_requests += 1
                # Keep traceability with current request id.
                cached.request_id = request.request_id
                self._log("info", "Cache hit", request_id=request.request_id, provider=cached.provider, model=cached.model)
                self._record_request_metrics(request, cached, start_time, complexity, routing_strategy)
                if self.observability:
                    self.observability.on_request_end(
                        provider=cached.provider,
                        model=cached.model,
                        success=cached.success,
                        latency_ms=cached.latency_ms,
                        cached=cached.cached,
                        fallback_used=cached.fallback_used,
                        error_message=cached.error_message,
                        request_span=request_span,
                    )
                return cached
            if self.observability:
                self.observability.on_cache_lookup(hit=False)
            self._cache_misses += 1

        # Primary provider
        primary = self._router.select_provider(request, routing_strategy)
        if not primary:
            self._failed_requests += 1
            response = LLMResponse(
                content="",
                provider="none",
                model="",
                request_id=request.request_id,
                success=False,
                error_message="No healthy providers available",
                latency_ms=(time.time() - start_time) * 1000,
            )
            self._record_request_metrics(request, response, start_time, complexity, routing_strategy)
            if self.observability:
                self.observability.on_request_end(
                    provider=response.provider,
                    model=response.model,
                    success=response.success,
                    latency_ms=response.latency_ms,
                    cached=response.cached,
                    fallback_used=response.fallback_used,
                    error_message=response.error_message,
                    request_span=request_span,
                )
            return response

        attempted = [primary.name]
        response = await self._execute_provider(primary, request)
        if self.observability:
            self.observability.on_provider_attempt(
                provider=primary.name,
                model=response.model,
                success=response.success,
                latency_ms=response.latency_ms,
                error_message=response.error_message,
                request_span=request_span,
            )

        if response.success:
            self._successful_requests += 1
            if self._cache:
                self._cache.set(request, response)
            self._log("info", "Primary provider succeeded", request_id=request.request_id, provider=response.provider, model=response.model)
            self._record_request_metrics(request, response, start_time, complexity, routing_strategy)
            if self.observability:
                self.observability.on_request_end(
                    provider=response.provider,
                    model=response.model,
                    success=response.success,
                    latency_ms=response.latency_ms,
                    cached=response.cached,
                    fallback_used=response.fallback_used,
                    error_message=response.error_message,
                    request_span=request_span,
                )
            return response

        self._log(
            "warning",
            "Primary provider failed",
            request_id=request.request_id,
            provider=primary.name,
            error=response.error_message,
        )

        # Fallback providers
        fallbacks = self._router.get_fallback_providers(primary, request)
        for index, fallback in enumerate(fallbacks):
            if index >= self._max_fallback_attempts:
                break

            self._fallback_uses += 1
            attempted.append(fallback.name)

            fallback_response = await self._execute_provider(fallback, request)
            if self.observability:
                self.observability.on_provider_attempt(
                    provider=fallback.name,
                    model=fallback_response.model,
                    success=fallback_response.success,
                    latency_ms=fallback_response.latency_ms,
                    error_message=fallback_response.error_message,
                    request_span=request_span,
                )
            if fallback_response.success:
                self._successful_requests += 1
                fallback_response.fallback_used = True
                fallback_response.attempted_providers = attempted
                if self._cache:
                    self._cache.set(request, fallback_response)
                self._log("info", "Fallback provider succeeded", request_id=request.request_id, provider=fallback_response.provider, model=fallback_response.model)
                self._record_request_metrics(request, fallback_response, start_time, complexity, routing_strategy)
                if self.observability:
                    self.observability.on_request_end(
                        provider=fallback_response.provider,
                        model=fallback_response.model,
                        success=fallback_response.success,
                        latency_ms=fallback_response.latency_ms,
                        cached=fallback_response.cached,
                        fallback_used=fallback_response.fallback_used,
                        error_message=fallback_response.error_message,
                        request_span=request_span,
                    )
                return fallback_response

        self._failed_requests += 1
        response = LLMResponse(
            content="",
            provider="none",
            model="",
            request_id=request.request_id,
            success=False,
            error_message=response.error_message or "All providers failed",
            latency_ms=(time.time() - start_time) * 1000,
            attempted_providers=attempted,
        )
        self._record_request_metrics(request, response, start_time, complexity, routing_strategy)
        if self.observability:
            self.observability.on_request_end(
                provider=response.provider,
                model=response.model,
                success=response.success,
                latency_ms=response.latency_ms,
                cached=response.cached,
                fallback_used=response.fallback_used,
                error_message=response.error_message,
                request_span=request_span,
            )
        return response

    def get_stats(self) -> Dict[str, Any]:
        total = self._total_requests
        cache_hit_rate = (self._cache_hits / total * 100) if total else 0.0
        success_rate = (self._successful_requests / total * 100) if total else 0.0

        stats: Dict[str, Any] = {
            "total_requests": self._total_requests,
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
            "success_rate_percent": round(success_rate, 2),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "fallback_uses": self._fallback_uses,
            "providers": {},
        }

        if self._cache:
            stats["cache_stats"] = self._cache.get_stats()

        if self.metrics_collector:
            stats["observability"] = self.metrics_collector.get_stats()
        if self.observability:
            stats["telemetry"] = self.observability.get_status()

        if self.circuit_breakers:
            stats["circuit_breakers"] = {
                name: breaker.get_status()
                for name, breaker in self.circuit_breakers.items()
            }

        for provider in self._providers:
            metrics = provider.get_metrics()
            stats["providers"][provider.name] = {
                "status": metrics.status.value,
                "success_rate": round(metrics.success_rate * 100, 2),
                "avg_latency_ms": round(metrics.avg_latency_ms, 2),
                "total_requests": metrics.total_requests,
                "error_count": metrics.error_count,
            }

        return stats

    def get_healthy_providers(self) -> List[str]:
        return [p.name for p in self._providers if p.is_healthy()]

    def health(self) -> Dict[str, Any]:
        """Return service health based on provider runtime status."""
        healthy_providers = self.get_healthy_providers()
        total = len(self._providers)

        if total == 0 or not healthy_providers:
            status = "unhealthy"
        elif len(healthy_providers) < total:
            status = "degraded"
        else:
            status = "healthy"

        return {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "healthy_providers": healthy_providers,
            "total_providers": total,
            "stats": {
                "total_requests": self._total_requests,
                "success_rate_percent": round(
                    (self._successful_requests / self._total_requests * 100)
                    if self._total_requests
                    else 0.0,
                    2,
                ),
            },
        }

    def readiness(self) -> Dict[str, Any]:
        """Return readiness checks for dependencies and providers."""
        provider_checks: Dict[str, Dict[str, Any]] = {}
        ready = True

        for provider in self._providers:
            configured = provider.validate_config()
            healthy = provider.is_healthy()
            provider_checks[provider.name] = {
                "configured": configured,
                "healthy": healthy,
            }
            if not configured:
                ready = False

        cache_ready = True
        if self._cache:
            try:
                self._cache.get_stats()
            except Exception:
                cache_ready = False
                ready = False

        if not self._providers:
            ready = False

        return {
            "ready": ready,
            "timestamp": datetime.now().isoformat(),
            "providers": provider_checks,
            "cache": {
                "enabled": self._cache is not None,
                "ready": cache_ready,
            },
            "telemetry": self.observability.get_status() if self.observability else {
                "prometheus_enabled": False,
                "otel_enabled": False,
            },
        }

    def clear_cache(self) -> None:
        if self._cache:
            self._cache.clear()

    def reset_metrics(self) -> None:
        self._total_requests = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._fallback_uses = 0
        for provider in self._providers:
            provider.reset_metrics()

    def set_routing_strategy(self, strategy: RoutingStrategy) -> None:
        self._router.set_strategy(strategy)

    def add_provider(self, provider: BaseProvider) -> None:
        if provider not in self._providers:
            self._providers.append(provider)
            self._router.add_provider(provider)

    def remove_provider(self, provider_name: str) -> None:
        self._providers = [p for p in self._providers if p.name != provider_name]
        self._router.remove_provider(provider_name)

    async def close(self) -> None:
        """Close provider resources and release network clients."""
        for provider in self._providers:
            close_fn = getattr(provider, "close", None)
            if callable(close_fn):
                result = close_fn()
                if inspect.isawaitable(result):
                    await result
        if self.observability:
            close_fn = getattr(self.observability, "close", None)
            if callable(close_fn):
                result = close_fn()
                if inspect.isawaitable(result):
                    await result
