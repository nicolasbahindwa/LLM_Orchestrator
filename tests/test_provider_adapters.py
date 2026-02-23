import asyncio

from llm_orchestrator.config.loader import load_from_dict
from llm_orchestrator.core import (
    AdapterResult,
    ProviderAdapter,
    create_provider_adapter,
    register_provider_adapter,
)
from llm_orchestrator.models import LLMRequest
from llm_orchestrator.provider_loader import load_providers


class TestCustomAdapter(ProviderAdapter):
    async def generate(self, *, client, request, model, provider_name, provider_config):
        _ = client
        _ = provider_name
        _ = provider_config
        return AdapterResult(
            content=f"custom:{request.prompt}",
            model=model,
            usage={"total_tokens": 1},
            raw_response={"ok": True},
        )


def test_create_provider_adapter_from_path() -> None:
    adapter = create_provider_adapter("tests.test_provider_adapters:TestCustomAdapter")
    assert isinstance(adapter, ProviderAdapter)
    assert adapter.__class__.__name__ == "TestCustomAdapter"


def test_register_provider_adapter() -> None:
    register_provider_adapter("test_custom", TestCustomAdapter)
    adapter = create_provider_adapter("test_custom")
    assert isinstance(adapter, TestCustomAdapter)


def test_load_providers_with_custom_adapter_path() -> None:
    config = load_from_dict(
        {
            "models": {
                "my-model": {
                    "provider": "my_provider",
                    "complexity_level": "moderate",
                    "cost_tier": "cheap",
                    "speed_tier": "fast",
                    "quality_tier": "good",
                }
            },
            "providers": {
                "my_provider": {
                    "enabled": True,
                    "adapter": "tests.test_provider_adapters:TestCustomAdapter",
                    "default_model": "my-model",
                }
            },
        }
    )

    providers = load_providers(config)
    assert len(providers) == 1

    response = asyncio.run(providers[0].generate(LLMRequest(prompt="hello")))
    assert response.success is True
    assert response.content == "custom:hello"
    assert response.model == "my-model"
