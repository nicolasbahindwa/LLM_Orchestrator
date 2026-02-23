from llm_orchestrator.config.loader import load_from_dict, load_from_env, substitute_env_vars


def test_substitute_env_vars_uses_default_when_missing() -> None:
    out = substitute_env_vars("token=${MISSING_VAR:abc123}")
    assert out == "token=abc123"


def test_load_from_dict_parses_model_provider_and_cache() -> None:
    config = load_from_dict(
        {
            "models": {
                "gpt-4o-mini": {
                    "provider": "openai",
                    "complexity_level": "moderate",
                    "cost_tier": "cheap",
                    "speed_tier": "fast",
                    "quality_tier": "good",
                }
            },
            "providers": {
                "openai": {
                    "enabled": "true",
                    "adapter": "openai_compatible",
                    "adapter_config": {"chat_completions_path": "/chat/completions"},
                    "api_key": "sk-test",
                    "base_url": "https://api.openai.com/v1",
                    "default_model": "gpt-4o-mini",
                }
            },
            "cache": {
                "enabled": "true",
                "ttl": "120",
            },
        }
    )

    assert "openai" in config.providers
    assert config.providers["openai"].enabled is True
    assert config.providers["openai"].adapter == "openai_compatible"
    assert config.providers["openai"].adapter_config["chat_completions_path"] == "/chat/completions"
    assert config.cache.ttl == 120
    assert config.get_enabled_models()[0].name == "gpt-4o-mini"


def test_load_from_env_handles_provider_suffixes_without_index_errors(monkeypatch) -> None:
    monkeypatch.setenv("LLM_ORCHESTRATOR_OPENAI_ENABLED", "true")
    monkeypatch.setenv("LLM_ORCHESTRATOR_OPENAI_API_KEY", "sk-from-env")
    monkeypatch.setenv("LLM_ORCHESTRATOR_OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("LLM_ORCHESTRATOR_ROUTING_STRATEGY", "speed")
    monkeypatch.setenv("LLM_ORCHESTRATOR_CACHE_TTL", "90")
    monkeypatch.setenv("LLM_ORCHESTRATOR_CACHE_BACKEND", "memory")

    config = load_from_env()

    assert "openai" in config.providers
    assert config.providers["openai"].enabled is True
    assert config.providers["openai"].api_key == "sk-from-env"
    assert config.routing.strategy.value == "speed"
    assert config.cache.ttl == 90
    assert config.cache.backend == "memory"


def test_load_from_dict_parses_extended_observability_settings() -> None:
    config = load_from_dict(
        {
            "monitoring": {
                "metrics": {
                    "enabled": True,
                    "prometheus_enabled": True,
                    "prometheus_host": "127.0.0.1",
                    "prometheus_port": 9477,
                    "prometheus_start_http_server": False,
                    "otel_enabled": True,
                    "otel_service_name": "my-service",
                    "otel_exporter_endpoint": "http://localhost:4318/v1/traces",
                    "otel_exporter_insecure": False,
                    "sinks": ["my_project.observability:DatadogSink"],
                }
            }
        }
    )

    metrics = config.monitoring.metrics
    assert metrics.prometheus_enabled is True
    assert metrics.prometheus_host == "127.0.0.1"
    assert metrics.prometheus_port == 9477
    assert metrics.prometheus_start_http_server is False
    assert metrics.otel_enabled is True
    assert metrics.otel_service_name == "my-service"
    assert metrics.otel_exporter_endpoint == "http://localhost:4318/v1/traces"
    assert metrics.otel_exporter_insecure is False
    assert metrics.sinks == ["my_project.observability:DatadogSink"]


def test_load_from_env_parses_monitoring_metrics_settings(monkeypatch) -> None:
    monkeypatch.setenv("LLM_ORCHESTRATOR_MONITORING_METRICS_ENABLED", "true")
    monkeypatch.setenv("LLM_ORCHESTRATOR_MONITORING_METRICS_PROMETHEUS_ENABLED", "true")
    monkeypatch.setenv("LLM_ORCHESTRATOR_MONITORING_METRICS_PROMETHEUS_PORT", "9488")
    monkeypatch.setenv("LLM_ORCHESTRATOR_MONITORING_METRICS_OTEL_ENABLED", "true")
    monkeypatch.setenv("LLM_ORCHESTRATOR_MONITORING_METRICS_OTEL_SERVICE_NAME", "svc-a")
    monkeypatch.setenv("LLM_ORCHESTRATOR_MONITORING_METRICS_SINKS", "my.mod:SinkA,my.mod:SinkB")
    monkeypatch.setenv(
        "LLM_ORCHESTRATOR_MONITORING_METRICS_OTEL_EXPORTER_ENDPOINT",
        "http://localhost:4318/v1/traces",
    )

    config = load_from_env()

    assert config.monitoring.metrics.enabled is True
    assert config.monitoring.metrics.prometheus_enabled is True
    assert config.monitoring.metrics.prometheus_port == 9488
    assert config.monitoring.metrics.otel_enabled is True
    assert config.monitoring.metrics.otel_service_name == "svc-a"
    assert config.monitoring.metrics.sinks == ["my.mod:SinkA", "my.mod:SinkB"]
    assert (
        config.monitoring.metrics.otel_exporter_endpoint
        == "http://localhost:4318/v1/traces"
    )


def test_load_from_yaml_empty_file_returns_default_config(tmp_path) -> None:
    from llm_orchestrator.config.loader import load_from_yaml

    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    config = load_from_yaml(str(path))

    assert config is not None
    assert config.models == {}
    assert config.providers == {}
