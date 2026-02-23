"""Single orchestrator factory.

This module is the only creation path for llm_orchestrator.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .cache import LLMCache, MemoryCache, RedisCache
from .config import OrchestratorConfig, load_config
from .core import Orchestrator
from .fallbacks import CircuitBreakerConfig as RuntimeCircuitBreakerConfig
from .fallbacks import CircuitBreakerRegistry
from .fallbacks.retry_handler import RetryConfig, RetryHandler, RetryStrategy
from .monitoring import MetricsCollector, ObservabilityManager, StructuredLogger
from .monitoring.logger import LogFormat
from .provider_loader import load_providers
from .routing import ModelRouter, Router, RoutingStrategy
from .utils import ComplexityAnalyzer


def create_orchestrator(
    file_path: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
    from_env: bool = False,
    config: Optional[OrchestratorConfig] = None,
) -> Orchestrator:
    """Create orchestrator from YAML/JSON/dict/env config."""
    if config is None:
        config = load_config(file_path=file_path, data=data, from_env=from_env)

    providers = load_providers(config)
    if not providers:
        raise ValueError(
            "No valid providers were loaded. "
            "Check your YAML providers section and API keys."
        )

    cache = None
    if config.cache.enabled:
        # Select cache backend based on configuration
        if config.cache.backend == "redis":
            backend = RedisCache(
                host=config.cache.redis_host,
                port=config.cache.redis_port,
                db=config.cache.redis_db,
                password=config.cache.redis_password,
                username=config.cache.redis_username,
                prefix=config.cache.redis_prefix,
                max_connections=config.cache.redis_max_connections,
                socket_timeout=config.cache.redis_socket_timeout,
            )
        else:
            # Default to memory cache
            backend = MemoryCache(max_size=config.cache.max_size)

        cache = LLMCache(
            backend=backend,
            ttl_seconds=config.cache.ttl,
        )

    router = Router(
        providers=providers,
        strategy=RoutingStrategy(config.routing.strategy.value),
        fallback_strategy=config.routing.fallback_strategy,
    )

    orchestrator = Orchestrator(
        providers=providers,
        cache=cache,
        router=router,
        max_fallback_attempts=config.routing.max_fallback_attempts,
    )

    if config.resilience.retry.enabled:
        orchestrator.retry_handler = RetryHandler(
            RetryConfig(
                max_retries=config.resilience.retry.max_retries,
                base_delay_seconds=config.resilience.retry.base_delay,
                max_delay_seconds=config.resilience.retry.max_delay,
                strategy=RetryStrategy(config.resilience.retry.strategy.value),
            )
        )

    if config.resilience.circuit_breaker.enabled:
        registry = CircuitBreakerRegistry()
        breaker_config = RuntimeCircuitBreakerConfig(
            failure_threshold=config.resilience.circuit_breaker.failure_threshold,
            recovery_timeout_seconds=config.resilience.circuit_breaker.recovery_timeout,
        )
        orchestrator.circuit_breakers = {
            provider.name: registry.get_breaker(provider.name, config=breaker_config)
            for provider in providers
        }

    if config.monitoring.logging.enabled:
        level_name = str(config.monitoring.logging.level).upper()
        level = getattr(logging, level_name, logging.INFO)
        fmt = LogFormat.JSON if str(config.monitoring.logging.format).lower() == "json" else LogFormat.TEXT

        output_file = None
        if config.monitoring.logging.output in {"file", "both"}:
            output_file = config.monitoring.logging.file_path

        orchestrator.logger = StructuredLogger(
            name="llm_orchestrator",
            level=level,
            format=fmt,
            output_file=output_file,
        )

    if config.monitoring.metrics.enabled:
        orchestrator.metrics_collector = MetricsCollector()
        orchestrator.observability = ObservabilityManager(
            prometheus_enabled=config.monitoring.metrics.prometheus_enabled,
            prometheus_host=config.monitoring.metrics.prometheus_host,
            prometheus_port=config.monitoring.metrics.prometheus_port,
            prometheus_start_http_server=config.monitoring.metrics.prometheus_start_http_server,
            otel_enabled=config.monitoring.metrics.otel_enabled,
            otel_service_name=config.monitoring.metrics.otel_service_name,
            otel_exporter_endpoint=config.monitoring.metrics.otel_exporter_endpoint,
            otel_exporter_insecure=config.monitoring.metrics.otel_exporter_insecure,
            sinks=list(config.monitoring.metrics.sinks),
        )

    if config.routing.complexity_routing.enabled:
        orchestrator.complexity_analyzer = ComplexityAnalyzer()
        orchestrator.model_router = ModelRouter(config)

    orchestrator.config = config
    return orchestrator


def create_orchestrator_from_yaml(file_path: str) -> Orchestrator:
    return create_orchestrator(file_path=file_path)


def create_orchestrator_from_json(file_path: str) -> Orchestrator:
    return create_orchestrator(file_path=file_path)


def create_orchestrator_from_dict(data: Dict[str, Any]) -> Orchestrator:
    return create_orchestrator(data=data)


def create_orchestrator_from_env() -> Orchestrator:
    return create_orchestrator(from_env=True)
