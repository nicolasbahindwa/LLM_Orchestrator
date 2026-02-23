from llm_orchestrator.monitoring import ObservabilityManager


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[str] = []

    def on_request_start(self, **kwargs):  # type: ignore[no-untyped-def]
        _ = kwargs
        self.events.append("start")
        return "ctx"

    def on_cache_lookup(self, *, hit: bool) -> None:
        self.events.append(f"cache:{hit}")

    def on_provider_attempt(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        _ = kwargs
        self.events.append("provider")

    def on_request_end(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        _ = kwargs
        self.events.append("end")

    def close(self) -> None:
        self.events.append("close")


def test_observability_manager_noop_and_status() -> None:
    manager = ObservabilityManager(
        prometheus_enabled=True,
        prometheus_start_http_server=False,
        otel_enabled=True,
    )

    span = manager.on_request_start(
        request_id="req-1",
        routing_strategy="balanced",
        task_complexity="moderate",
    )
    manager.on_cache_lookup(hit=False)
    manager.on_provider_attempt(
        provider="openai",
        success=True,
        latency_ms=12.5,
        model="gpt-4o-mini",
        request_span=span,
    )
    manager.on_request_end(
        provider="openai",
        model="gpt-4o-mini",
        success=True,
        latency_ms=15.1,
        cached=False,
        fallback_used=False,
        request_span=span,
    )

    status = manager.get_status()
    assert "prometheus_enabled" in status
    assert "otel_enabled" in status
    assert "prometheus_available" in status
    assert "otel_api_available" in status

    manager.close()


def test_observability_manager_loads_custom_sink_from_path() -> None:
    manager = ObservabilityManager(
        prometheus_enabled=False,
        otel_enabled=False,
        sinks=["tests.test_observability:RecordingSink"],
    )

    span = manager.on_request_start(
        request_id="req-2",
        routing_strategy="balanced",
        task_complexity="simple",
    )
    manager.on_cache_lookup(hit=True)
    manager.on_provider_attempt(
        provider="openai",
        success=True,
        latency_ms=5.0,
        request_span=span,
    )
    manager.on_request_end(
        provider="openai",
        model="m",
        success=True,
        latency_ms=10.0,
        cached=True,
        fallback_used=False,
        request_span=span,
    )
    manager.close()

    status = manager.get_status()
    assert status["custom_sinks_errors"] == {}
    assert "RecordingSink" in status["custom_sinks_loaded"]
