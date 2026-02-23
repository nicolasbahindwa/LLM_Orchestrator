from llm_orchestrator.models import LLMRequest
from llm_orchestrator.routing import Router, RoutingStrategy

from .helpers import FakeProvider, set_provider_latency


def _request() -> LLMRequest:
    return LLMRequest(prompt="route this")


def test_router_cost_prefers_google() -> None:
    providers = [FakeProvider("openai"), FakeProvider("anthropic"), FakeProvider("google")]
    router = Router(providers, strategy=RoutingStrategy.COST)

    selected = router.select_provider(_request())

    assert selected is not None
    assert selected.name == "google"


def test_router_quality_prefers_anthropic() -> None:
    providers = [FakeProvider("openai"), FakeProvider("anthropic"), FakeProvider("google")]
    router = Router(providers, strategy=RoutingStrategy.QUALITY)

    selected = router.select_provider(_request())

    assert selected is not None
    assert selected.name == "anthropic"


def test_router_speed_uses_lowest_latency() -> None:
    openai = FakeProvider("openai")
    anthropic = FakeProvider("anthropic")
    google = FakeProvider("google")

    set_provider_latency(openai, 300)
    set_provider_latency(anthropic, 120)
    set_provider_latency(google, 220)

    router = Router([openai, anthropic, google], strategy=RoutingStrategy.SPEED)
    selected = router.select_provider(_request())

    assert selected is not None
    assert selected.name == "anthropic"


def test_router_round_robin_rotates() -> None:
    p1 = FakeProvider("p1")
    p2 = FakeProvider("p2")
    p3 = FakeProvider("p3")
    router = Router([p1, p2, p3], strategy=RoutingStrategy.ROUND_ROBIN)

    picks = [router.select_provider(_request()).name for _ in range(4)]

    assert picks == ["p1", "p2", "p3", "p1"]


def test_router_returns_none_if_no_healthy_providers() -> None:
    router = Router([FakeProvider("bad", healthy=False)])
    assert router.select_provider(_request()) is None


def test_router_honors_provider_hint() -> None:
    providers = [FakeProvider("openai"), FakeProvider("anthropic"), FakeProvider("google")]
    router = Router(providers, strategy=RoutingStrategy.COST)
    request = LLMRequest(prompt="route this", metadata={"provider": "anthropic"})

    selected = router.select_provider(request)

    assert selected is not None
    assert selected.name == "anthropic"


def test_router_fallback_cheapest_first() -> None:
    openai = FakeProvider("openai")
    anthropic = FakeProvider("anthropic")
    google = FakeProvider("google")

    router = Router(
        [openai, anthropic, google],
        strategy=RoutingStrategy.BALANCED,
        fallback_strategy="cheapest_first",
    )

    fallbacks = router.get_fallback_providers(anthropic, _request())

    assert [p.name for p in fallbacks] == ["google", "openai"]
