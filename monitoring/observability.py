"""Optional observability integrations for Prometheus/OpenTelemetry/custom sinks."""

from __future__ import annotations

import importlib
import threading
from typing import Any, Optional

from .sink import ObservabilitySink

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on optional deps
    Counter = Gauge = Histogram = None  # type: ignore[assignment]
    start_http_server = None  # type: ignore[assignment]
    PROMETHEUS_AVAILABLE = False

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode, set_span_in_context

    OTEL_API_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on optional deps
    trace = None  # type: ignore[assignment]
    Status = StatusCode = set_span_in_context = None  # type: ignore[assignment]
    OTEL_API_AVAILABLE = False

try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    OTEL_SDK_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on optional deps
    OTLPSpanExporter = Resource = TracerProvider = BatchSpanProcessor = None  # type: ignore[assignment]
    OTEL_SDK_AVAILABLE = False


_PROM_LOCK = threading.Lock()
_PROM_METRICS_INITIALIZED = False
_PROM_SERVER_STARTED = False

_REQUEST_COUNTER: Any = None
_REQUEST_LATENCY_MS: Any = None
_PROVIDER_ATTEMPT_COUNTER: Any = None
_PROVIDER_LATENCY_MS: Any = None
_CACHE_EVENTS_COUNTER: Any = None
_INFLIGHT_REQUESTS: Any = None


def _initialize_prometheus_metrics() -> None:
    """Initialize Prometheus metrics only once per process."""
    global _PROM_METRICS_INITIALIZED
    global _REQUEST_COUNTER
    global _REQUEST_LATENCY_MS
    global _PROVIDER_ATTEMPT_COUNTER
    global _PROVIDER_LATENCY_MS
    global _CACHE_EVENTS_COUNTER
    global _INFLIGHT_REQUESTS

    with _PROM_LOCK:
        if _PROM_METRICS_INITIALIZED:
            return

        _REQUEST_COUNTER = Counter(
            "llm_orchestrator_requests_total",
            "Total orchestrated requests",
            ["provider", "model", "success", "cached", "fallback_used"],
        )
        _REQUEST_LATENCY_MS = Histogram(
            "llm_orchestrator_request_latency_ms",
            "Request latency in milliseconds",
            ["provider", "model", "success"],
        )
        _PROVIDER_ATTEMPT_COUNTER = Counter(
            "llm_orchestrator_provider_attempts_total",
            "Provider call attempts",
            ["provider", "success"],
        )
        _PROVIDER_LATENCY_MS = Histogram(
            "llm_orchestrator_provider_latency_ms",
            "Provider call latency in milliseconds",
            ["provider", "success"],
        )
        _CACHE_EVENTS_COUNTER = Counter(
            "llm_orchestrator_cache_events_total",
            "Cache events",
            ["result"],
        )
        _INFLIGHT_REQUESTS = Gauge(
            "llm_orchestrator_inflight_requests",
            "Current in-flight requests",
        )
        _PROM_METRICS_INITIALIZED = True


def _load_object(path: str) -> Any:
    """Load object from 'module:Name' or 'module.Name'."""
    if ":" in path:
        module_name, object_name = path.split(":", 1)
    else:
        module_name, _, object_name = path.rpartition(".")
        if not module_name:
            raise ValueError(f"Invalid class path: {path}")
    module = importlib.import_module(module_name)
    loaded = getattr(module, object_name, None)
    if loaded is None:
        raise ValueError(f"Object '{object_name}' not found in module '{module_name}'")
    return loaded


class ObservabilityManager:
    """Best-effort telemetry manager with pluggable custom sinks."""

    def __init__(
        self,
        *,
        prometheus_enabled: bool = False,
        prometheus_host: str = "0.0.0.0",
        prometheus_port: int = 9464,
        prometheus_start_http_server: bool = True,
        otel_enabled: bool = False,
        otel_service_name: str = "llm-orchestrator",
        otel_exporter_endpoint: Optional[str] = None,
        otel_exporter_insecure: bool = True,
        sinks: Optional[list[str]] = None,
    ) -> None:
        self.prometheus_enabled = False
        self.otel_enabled = False
        self._otel_service_name = otel_service_name
        self._tracer: Any = None
        self._otel_provider: Any = None
        self._otel_error: Optional[str] = None
        self._prometheus_error: Optional[str] = None

        self._sinks: list[ObservabilitySink] = []
        self._sink_errors: dict[str, str] = {}

        if prometheus_enabled:
            self._setup_prometheus(
                host=prometheus_host,
                port=prometheus_port,
                start_server=prometheus_start_http_server,
            )

        if otel_enabled:
            self._setup_otel(
                service_name=otel_service_name,
                exporter_endpoint=otel_exporter_endpoint,
                exporter_insecure=otel_exporter_insecure,
            )

        for sink_path in sinks or []:
            self.add_sink_from_path(sink_path)

    def add_sink(self, sink: ObservabilitySink) -> None:
        """Register a custom sink instance."""
        self._sinks.append(sink)

    def add_sink_from_path(self, sink_path: str) -> None:
        """Load and register sink by import path."""
        try:
            loaded = _load_object(sink_path)
            sink_instance: Any
            if isinstance(loaded, type):
                sink_instance = loaded()
            elif callable(loaded):
                sink_instance = loaded()
            else:
                sink_instance = loaded

            required_methods = [
                "on_request_start",
                "on_cache_lookup",
                "on_provider_attempt",
                "on_request_end",
            ]
            for method_name in required_methods:
                if not callable(getattr(sink_instance, method_name, None)):
                    raise ValueError(
                        f"Sink '{sink_path}' is missing method '{method_name}'"
                    )

            self._sinks.append(sink_instance)
        except Exception as exc:
            self._sink_errors[sink_path] = str(exc)

    def _setup_prometheus(self, *, host: str, port: int, start_server: bool) -> None:
        global _PROM_SERVER_STARTED

        if not PROMETHEUS_AVAILABLE:
            self._prometheus_error = (
                "prometheus-client is not installed. "
                "Install with: pip install llm-orchestrator[observability]"
            )
            return

        try:
            _initialize_prometheus_metrics()
            self.prometheus_enabled = True
            if start_server and not _PROM_SERVER_STARTED:
                start_http_server(port=port, addr=host)
                _PROM_SERVER_STARTED = True
        except Exception as exc:  # pragma: no cover - defensive runtime path
            self._prometheus_error = str(exc)
            self.prometheus_enabled = False

    def _setup_otel(
        self,
        *,
        service_name: str,
        exporter_endpoint: Optional[str],
        exporter_insecure: bool,
    ) -> None:
        if not OTEL_API_AVAILABLE:
            self._otel_error = (
                "opentelemetry-api is not installed. "
                "Install with: pip install llm-orchestrator[observability]"
            )
            return

        try:
            if exporter_endpoint and OTEL_SDK_AVAILABLE:
                current_provider = trace.get_tracer_provider()
                if current_provider.__class__.__name__ == "ProxyTracerProvider":
                    resource = Resource.create({"service.name": service_name})
                    tracer_provider = TracerProvider(resource=resource)
                    exporter = OTLPSpanExporter(endpoint=exporter_endpoint)
                    tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
                    trace.set_tracer_provider(tracer_provider)
                    self._otel_provider = tracer_provider

            _ = exporter_insecure
            self._tracer = trace.get_tracer("llm_orchestrator", "0.2.0")
            self.otel_enabled = True
        except Exception as exc:  # pragma: no cover - defensive runtime path
            self._otel_error = str(exc)
            self.otel_enabled = False

    def _root_span_from_context(self, request_span: Optional[Any]) -> Optional[Any]:
        if isinstance(request_span, dict):
            return request_span.get("root_span")
        return request_span

    def _sink_contexts_from_context(self, request_span: Optional[Any]) -> list[tuple[Any, Any]]:
        if isinstance(request_span, dict):
            contexts = request_span.get("sink_contexts", [])
            if isinstance(contexts, list):
                return contexts
        return []

    def on_request_start(
        self,
        *,
        request_id: str,
        routing_strategy: str,
        task_complexity: str = "",
        provider_hint: str = "",
        model_hint: str = "",
    ) -> Optional[Any]:
        """Start request telemetry and return opaque context object."""
        if self.prometheus_enabled and _INFLIGHT_REQUESTS is not None:
            _INFLIGHT_REQUESTS.inc()

        root_span = None
        if self.otel_enabled and self._tracer is not None:
            root_span = self._tracer.start_span("llm.request")
            root_span.set_attribute("llm.request_id", request_id)
            root_span.set_attribute("llm.routing_strategy", routing_strategy or "unknown")
            root_span.set_attribute("llm.task_complexity", task_complexity or "unknown")
            if provider_hint:
                root_span.set_attribute("llm.provider_hint", provider_hint)
            if model_hint:
                root_span.set_attribute("llm.model_hint", model_hint)

        sink_contexts: list[tuple[Any, Any]] = []
        for sink in self._sinks:
            try:
                sink_ctx = sink.on_request_start(
                    request_id=request_id,
                    routing_strategy=routing_strategy,
                    task_complexity=task_complexity,
                    provider_hint=provider_hint,
                    model_hint=model_hint,
                )
                sink_contexts.append((sink, sink_ctx))
            except Exception:
                continue

        if root_span is None and not sink_contexts:
            return None
        return {
            "root_span": root_span,
            "sink_contexts": sink_contexts,
        }

    def on_cache_lookup(self, *, hit: bool) -> None:
        """Track cache hit/miss."""
        if self.prometheus_enabled and _CACHE_EVENTS_COUNTER is not None:
            _CACHE_EVENTS_COUNTER.labels(result="hit" if hit else "miss").inc()

        for sink in self._sinks:
            try:
                sink.on_cache_lookup(hit=hit)
            except Exception:
                continue

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
        """Track provider attempt metrics/spans and fan-out to custom sinks."""
        success_label = str(bool(success)).lower()

        if self.prometheus_enabled:
            if _PROVIDER_ATTEMPT_COUNTER is not None:
                _PROVIDER_ATTEMPT_COUNTER.labels(
                    provider=provider or "unknown",
                    success=success_label,
                ).inc()
            if _PROVIDER_LATENCY_MS is not None:
                _PROVIDER_LATENCY_MS.labels(
                    provider=provider or "unknown",
                    success=success_label,
                ).observe(max(latency_ms, 0.0))

        root_span = self._root_span_from_context(request_span)
        if self.otel_enabled and self._tracer is not None:
            context = None
            if root_span is not None:
                context = set_span_in_context(root_span)
            span = self._tracer.start_span("llm.provider.attempt", context=context)
            span.set_attribute("llm.provider", provider or "unknown")
            span.set_attribute("llm.model", model or "unknown")
            span.set_attribute("llm.success", bool(success))
            span.set_attribute("llm.latency_ms", max(latency_ms, 0.0))
            if error_message:
                span.set_attribute("llm.error", error_message)
                span.set_status(Status(StatusCode.ERROR, error_message))
            else:
                span.set_status(Status(StatusCode.OK))
            span.end()

        context_by_sink = dict(self._sink_contexts_from_context(request_span))
        for sink in self._sinks:
            try:
                sink.on_provider_attempt(
                    provider=provider,
                    success=success,
                    latency_ms=latency_ms,
                    model=model,
                    error_message=error_message,
                    request_span=context_by_sink.get(sink),
                )
            except Exception:
                continue

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
        """Finalize request-level telemetry."""
        success_label = str(bool(success)).lower()
        cached_label = str(bool(cached)).lower()
        fallback_label = str(bool(fallback_used)).lower()

        if self.prometheus_enabled:
            if _REQUEST_COUNTER is not None:
                _REQUEST_COUNTER.labels(
                    provider=provider or "unknown",
                    model=model or "unknown",
                    success=success_label,
                    cached=cached_label,
                    fallback_used=fallback_label,
                ).inc()
            if _REQUEST_LATENCY_MS is not None:
                _REQUEST_LATENCY_MS.labels(
                    provider=provider or "unknown",
                    model=model or "unknown",
                    success=success_label,
                ).observe(max(latency_ms, 0.0))
            if _INFLIGHT_REQUESTS is not None:
                _INFLIGHT_REQUESTS.dec()

        root_span = self._root_span_from_context(request_span)
        if root_span is not None:
            root_span.set_attribute("llm.provider", provider or "unknown")
            root_span.set_attribute("llm.model", model or "unknown")
            root_span.set_attribute("llm.success", bool(success))
            root_span.set_attribute("llm.cached", bool(cached))
            root_span.set_attribute("llm.fallback_used", bool(fallback_used))
            root_span.set_attribute("llm.latency_ms", max(latency_ms, 0.0))

            if error_message:
                root_span.set_attribute("llm.error", error_message)
                root_span.set_status(Status(StatusCode.ERROR, error_message))
            else:
                root_span.set_status(Status(StatusCode.OK))
            root_span.end()

        context_by_sink = dict(self._sink_contexts_from_context(request_span))
        for sink in self._sinks:
            try:
                sink.on_request_end(
                    provider=provider,
                    model=model,
                    success=success,
                    latency_ms=latency_ms,
                    cached=cached,
                    fallback_used=fallback_used,
                    error_message=error_message,
                    request_span=context_by_sink.get(sink),
                )
            except Exception:
                continue

    def get_status(self) -> dict[str, Any]:
        """Return runtime status of configured telemetry backends."""
        return {
            "prometheus_enabled": self.prometheus_enabled,
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "prometheus_error": self._prometheus_error,
            "otel_enabled": self.otel_enabled,
            "otel_api_available": OTEL_API_AVAILABLE,
            "otel_sdk_available": OTEL_SDK_AVAILABLE,
            "otel_service_name": self._otel_service_name,
            "otel_error": self._otel_error,
            "custom_sinks_loaded": [sink.__class__.__name__ for sink in self._sinks],
            "custom_sinks_errors": dict(self._sink_errors),
        }

    def close(self) -> None:
        """Flush sinks and OTel providers."""
        for sink in self._sinks:
            try:
                sink.close()
            except Exception:
                continue

        if self._otel_provider is None:
            return
        shutdown = getattr(self._otel_provider, "shutdown", None)
        if callable(shutdown):
            shutdown()
