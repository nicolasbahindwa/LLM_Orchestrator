import pytest

from llm_orchestrator.models import LLMRequest, ProviderMetrics


def test_llm_request_defaults() -> None:
    req = LLMRequest(prompt="hello")
    assert req.max_tokens == 1000
    assert req.temperature == 0.7
    assert req.request_id.startswith("req_")


def test_llm_request_invalid_prompt_raises() -> None:
    with pytest.raises(ValueError):
        LLMRequest(prompt="   ")


def test_llm_request_invalid_max_tokens_raises() -> None:
    with pytest.raises(ValueError):
        LLMRequest(prompt="hello", max_tokens=0)


def test_llm_request_invalid_temperature_raises() -> None:
    with pytest.raises(ValueError):
        LLMRequest(prompt="hello", temperature=2.5)


def test_provider_metrics_updates() -> None:
    metrics = ProviderMetrics(provider_name="fake")
    metrics.update_success(120.0)
    metrics.update_error()

    assert metrics.total_requests == 2
    assert metrics.error_count == 1
    assert metrics.success_rate == 0.5
    assert metrics.last_success is not None
    assert metrics.last_error is not None