import pytest

from llm_orchestrator.config import OrchestratorConfig
from llm_orchestrator.factory import create_orchestrator

from .helpers import FakeProvider


def test_create_orchestrator_raises_when_no_providers(monkeypatch) -> None:
    import llm_orchestrator.factory as factory_module

    monkeypatch.setattr(factory_module, "load_providers", lambda _: [])

    with pytest.raises(ValueError, match="No valid providers were loaded"):
        create_orchestrator(config=OrchestratorConfig())


def test_create_orchestrator_wires_optional_components(monkeypatch) -> None:
    import llm_orchestrator.factory as factory_module

    monkeypatch.setattr(factory_module, "load_providers", lambda _: [FakeProvider("fake")])

    config = OrchestratorConfig()
    config.cache.enabled = True
    config.resilience.retry.enabled = True
    config.resilience.circuit_breaker.enabled = True
    config.monitoring.logging.enabled = True
    config.monitoring.metrics.enabled = True
    config.routing.complexity_routing.enabled = True

    orchestrator = create_orchestrator(config=config)

    assert orchestrator._cache is not None
    assert orchestrator.retry_handler is not None
    assert orchestrator.circuit_breakers
    assert orchestrator.logger is not None
    assert orchestrator.metrics_collector is not None
    assert orchestrator.observability is not None
    assert orchestrator.complexity_analyzer is not None
    assert orchestrator.model_router is not None


def test_create_orchestrator_respects_disabled_cache(monkeypatch) -> None:
    import llm_orchestrator.factory as factory_module

    monkeypatch.setattr(factory_module, "load_providers", lambda _: [FakeProvider("fake")])

    config = OrchestratorConfig()
    config.cache.enabled = False
    config.monitoring.metrics.enabled = False

    orchestrator = create_orchestrator(config=config)

    assert orchestrator._cache is None
    assert orchestrator.observability is None
