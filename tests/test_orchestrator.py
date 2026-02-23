import asyncio

from llm_orchestrator.cache import LLMCache, MemoryCache
from llm_orchestrator.core import Orchestrator
from llm_orchestrator.fallbacks import CircuitBreaker, CircuitBreakerConfig
from llm_orchestrator.monitoring import MetricsCollector
from llm_orchestrator.models import LLMRequest

from .helpers import FakeProvider


class DummyObservability:
    def __init__(self) -> None:
        self.cache_events: list[bool] = []
        self.provider_attempts: list[tuple[str, bool]] = []
        self.started = 0
        self.completed = 0

    def on_request_start(self, **kwargs):  # type: ignore[no-untyped-def]
        self.started += 1
        return None

    def on_cache_lookup(self, *, hit: bool) -> None:
        self.cache_events.append(hit)

    def on_provider_attempt(
        self,
        *,
        provider: str,
        success: bool,
        latency_ms: float,
        model: str = "",
        error_message: str | None = None,
        request_span=None,  # type: ignore[no-untyped-def]
    ) -> None:
        _ = latency_ms
        _ = model
        _ = error_message
        _ = request_span
        self.provider_attempts.append((provider, success))

    def on_request_end(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.completed += 1

    def get_status(self) -> dict[str, bool]:
        return {"prometheus_enabled": False, "otel_enabled": False}


def test_orchestrator_primary_success() -> None:
    provider = FakeProvider("primary")
    orchestrator = Orchestrator([provider])

    response = asyncio.run(orchestrator.generate(LLMRequest(prompt="hello")))

    assert response.success is True
    assert response.provider == "primary"
    stats = orchestrator.get_stats()
    assert stats["total_requests"] == 1
    assert stats["successful_requests"] == 1


def test_orchestrator_uses_fallback_when_primary_fails() -> None:
    primary = FakeProvider("primary", fail_times=1)
    fallback = FakeProvider("fallback")
    orchestrator = Orchestrator([primary, fallback], max_fallback_attempts=2)

    response = asyncio.run(orchestrator.generate(LLMRequest(prompt="need fallback")))

    assert response.success is True
    assert response.provider == "fallback"
    assert response.fallback_used is True
    assert response.attempted_providers == ["primary", "fallback"]


def test_orchestrator_returns_error_when_no_healthy_provider() -> None:
    bad = FakeProvider("bad", healthy=False)
    orchestrator = Orchestrator([bad])

    response = asyncio.run(orchestrator.generate(LLMRequest(prompt="hello")))

    assert response.success is False
    assert "No healthy providers available" in (response.error_message or "")


def test_orchestrator_cache_hit_skips_provider_call() -> None:
    provider = FakeProvider("primary")
    cache = LLMCache(backend=MemoryCache(), ttl_seconds=60)
    orchestrator = Orchestrator([provider], cache=cache)

    first = asyncio.run(orchestrator.generate(LLMRequest(prompt="cached prompt")))
    second = asyncio.run(orchestrator.generate(LLMRequest(prompt="cached prompt")))

    assert first.success is True
    assert second.success is True
    assert second.cached is True
    assert provider.calls == 1


def test_orchestrator_circuit_breaker_prevents_repeated_primary_calls() -> None:
    primary = FakeProvider("primary", fail_times=5)
    fallback = FakeProvider("fallback")
    orchestrator = Orchestrator([primary, fallback], max_fallback_attempts=1)
    orchestrator.circuit_breakers["primary"] = CircuitBreaker(
        "primary",
        CircuitBreakerConfig(failure_threshold=1, recovery_timeout_seconds=60),
    )

    first = asyncio.run(orchestrator.generate(LLMRequest(prompt="first")))
    second = asyncio.run(orchestrator.generate(LLMRequest(prompt="second")))

    assert first.success is True
    assert second.success is True
    assert primary.calls == 1
    assert fallback.calls == 2


def test_orchestrator_records_metrics_when_collector_enabled() -> None:
    provider = FakeProvider("primary")
    orchestrator = Orchestrator([provider])
    orchestrator.metrics_collector = MetricsCollector()

    asyncio.run(orchestrator.generate(LLMRequest(prompt="hello")))
    stats = orchestrator.get_stats()

    assert "observability" in stats
    assert stats["observability"]["total_requests"] == 1


def test_orchestrator_observability_hooks_are_called() -> None:
    provider = FakeProvider("primary")
    cache = LLMCache(backend=MemoryCache(), ttl_seconds=60)
    orchestrator = Orchestrator([provider], cache=cache)
    observer = DummyObservability()
    orchestrator.observability = observer

    asyncio.run(orchestrator.generate(LLMRequest(prompt="cache me")))
    asyncio.run(orchestrator.generate(LLMRequest(prompt="cache me")))

    assert observer.started == 2
    assert observer.completed == 2
    assert observer.cache_events == [False, True]
    assert observer.provider_attempts == [("primary", True)]


def test_orchestrator_health_and_readiness() -> None:
    healthy = FakeProvider("healthy", healthy=True)
    bad = FakeProvider("bad", healthy=False)
    orchestrator = Orchestrator([healthy, bad])

    health = orchestrator.health()
    readiness = orchestrator.readiness()

    assert health["status"] == "degraded"
    assert readiness["ready"] is False
    assert readiness["providers"]["healthy"]["configured"] is True
    assert readiness["providers"]["bad"]["configured"] is False
