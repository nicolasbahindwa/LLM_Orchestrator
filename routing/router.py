"""
Router - Intelligent provider selection.

This file decides WHICH provider to use for each request.

Why router exists:
- Choose fastest provider when user needs speed
- Choose cheapest provider when cost matters
- Choose most reliable provider for critical requests
- Balance load across providers

How it works:
1. Look at request requirements (priority, constraints)
2. Check which providers are healthy
3. Score each provider based on strategy
4. Pick the best one
"""

from typing import List, Optional, Dict, Any
from enum import Enum

from ..models import (
    LLMRequest,
    ProviderMetrics,
    RoutingContext,
    RequestPriority,
    ProviderStatus,
)
from ..core import BaseProvider


class RoutingStrategy(str, Enum):
    """
    Strategy for selecting providers.

    - COST: Always pick cheapest
    - QUALITY: Always pick highest quality (e.g., GPT-4, Claude Opus)
    - SPEED: Pick fastest (lowest latency)
    - BALANCED: Balance cost, quality, and speed
    - ROUND_ROBIN: Rotate between providers evenly
    """
    COST = "cost"
    QUALITY = "quality"
    SPEED = "speed"
    BALANCED = "balanced"
    ROUND_ROBIN = "round_robin"


class Router:
    """
    Intelligent router for selecting LLM providers.

    Makes smart decisions based on:
    - Routing strategy (cost/quality/speed/balanced)
    - Provider health (success rate, latency)
    - Request priority (critical vs normal)
    - User constraints (max cost, max latency)

    Example:
        router = Router(
            providers=[anthropic, openai, google],
            strategy=RoutingStrategy.BALANCED
        )

        # Router picks best provider
        provider = router.select_provider(request)
        response = await provider.generate(request)
    """

    def __init__(
        self,
        providers: List[BaseProvider],
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        fallback_strategy: str = "same_provider_first",
    ):
        """
        Initialize router.

        Args:
            providers: List of available providers
            strategy: Default routing strategy
        """
        self._providers = providers
        self._strategy = strategy
        self._fallback_strategy = fallback_strategy
        self._round_robin_index = 0

    def select_provider(
        self,
        request: LLMRequest,
        strategy: Optional[RoutingStrategy] = None,
    ) -> Optional[BaseProvider]:
        """
        Select best provider for a request.

        Process:
        1. Filter to healthy providers only
        2. Apply routing strategy to score each
        3. Return highest scoring provider

        Args:
            request: LLM request to route
            strategy: Override default strategy (optional)

        Returns:
            Best provider, or None if none available

        Example:
            provider = router.select_provider(
                request=LLMRequest(prompt="Hello"),
                strategy=RoutingStrategy.SPEED
            )
        """
        # Use provided strategy or default
        routing_strategy = strategy or self._strategy

        # Filter to healthy providers
        healthy_providers = [p for p in self._providers if p.is_healthy()]

        if not healthy_providers:
            return None

        # Respect explicit provider hint when present (e.g. model router decision).
        provider_hint = request.metadata.get("provider")
        if isinstance(provider_hint, str) and provider_hint.strip():
            hinted = [p for p in healthy_providers if p.name == provider_hint.strip()]
            if hinted:
                healthy_providers = hinted

        # Apply routing strategy
        if routing_strategy == RoutingStrategy.COST:
            return self._select_cheapest(healthy_providers, request)

        elif routing_strategy == RoutingStrategy.QUALITY:
            return self._select_highest_quality(healthy_providers, request)

        elif routing_strategy == RoutingStrategy.SPEED:
            return self._select_fastest(healthy_providers, request)

        elif routing_strategy == RoutingStrategy.BALANCED:
            return self._select_balanced(healthy_providers, request)

        elif routing_strategy == RoutingStrategy.ROUND_ROBIN:
            return self._select_round_robin(healthy_providers)

        else:
            # Fallback to first healthy provider
            return healthy_providers[0]

    def get_fallback_providers(
        self,
        primary_provider: BaseProvider,
        request: LLMRequest,
    ) -> List[BaseProvider]:
        """
        Get ordered list of fallback providers.

        Used when primary provider fails - try these next.

        Args:
            primary_provider: The provider that failed
            request: The request to route

        Returns:
            List of providers to try (ordered by preference)

        Example:
            primary = router.select_provider(request)
            try:
                response = await primary.generate(request)
            except:
                # Try fallbacks
                for fallback in router.get_fallback_providers(primary, request):
                    try:
                        response = await fallback.generate(request)
                        break
                    except:
                        continue
        """
        # Get all healthy providers except the failed one
        fallbacks = [
            p for p in self._providers
            if p.is_healthy() and p.name != primary_provider.name
        ]

        if self._fallback_strategy == "cheapest_first":
            # Lower cost rank first
            fallbacks.sort(key=lambda p: self._provider_cost_rank(p.name))
        elif self._fallback_strategy == "fastest_first":
            # Lower latency first
            fallbacks.sort(key=lambda p: p.get_metrics().avg_latency_ms)
        else:
            # Default: most reliable first
            fallbacks.sort(
                key=lambda p: p.get_metrics().success_rate,
                reverse=True
            )

        return fallbacks

    def _provider_cost_rank(self, provider_name: str) -> int:
        cost_order = {
            "google": 1,
            "anthropic": 2,
            "openai": 3,
        }
        return cost_order.get(provider_name, 99)

    # ══════════════════════════════════════════════════════════════════════
    # ROUTING STRATEGY IMPLEMENTATIONS
    # ══════════════════════════════════════════════════════════════════════

    def _select_cheapest(
        self,
        providers: List[BaseProvider],
        request: LLMRequest,
    ) -> BaseProvider:
        """
        Select cheapest provider.

        Note: In real implementation, would calculate cost based on:
        - Input tokens (from prompt)
        - Expected output tokens (from max_tokens)
        - Provider pricing

        For now, simple heuristic: Haiku < Sonnet < Opus

        Args:
            providers: Available providers
            request: Request to route

        Returns:
            Cheapest provider
        """
        # Simple cost heuristic by provider name
        # In production, would use actual pricing data
        cost_order = {
            "google": 1,      # Gemini often cheapest
            "anthropic": 2,   # Claude middle-tier
            "openai": 3,      # GPT-4 often most expensive
        }

        return min(
            providers,
            key=lambda p: cost_order.get(p.name, 99)
        )

    def _select_highest_quality(
        self,
        providers: List[BaseProvider],
        request: LLMRequest,
    ) -> BaseProvider:
        """
        Select highest quality provider.

        Quality heuristic:
        - Prefer Opus/GPT-4 for complex tasks
        - Prefer Sonnet/GPT-4o for balanced tasks
        - Consider success rate

        Args:
            providers: Available providers
            request: Request to route

        Returns:
            Highest quality provider
        """
        # Simple quality heuristic
        # In production, would analyze request complexity
        quality_order = {
            "anthropic": 3,  # Claude Opus/Sonnet high quality
            "openai": 2,     # GPT-4 high quality
            "google": 1,     # Gemini good quality
        }

        return max(
            providers,
            key=lambda p: quality_order.get(p.name, 0)
        )

    def _select_fastest(
        self,
        providers: List[BaseProvider],
        request: LLMRequest,
    ) -> BaseProvider:
        """
        Select fastest provider based on historical latency.

        Args:
            providers: Available providers
            request: Request to route

        Returns:
            Fastest provider
        """
        # Sort by average latency (lowest first)
        return min(
            providers,
            key=lambda p: p.get_metrics().avg_latency_ms
        )

    def _select_balanced(
        self,
        providers: List[BaseProvider],
        request: LLMRequest,
    ) -> BaseProvider:
        """
        Select provider with best balance of cost, quality, and speed.

        Scoring formula:
        score = (success_rate * 0.4) + (speed_score * 0.3) + (cost_score * 0.3)

        Higher success rate = better
        Lower latency = better
        Lower cost = better

        Args:
            providers: Available providers
            request: Request to route

        Returns:
            Best balanced provider
        """
        scores = []

        for provider in providers:
            metrics = provider.get_metrics()

            # Success rate score (0-1, higher is better)
            success_score = metrics.success_rate

            # Speed score (inverse of latency, normalized)
            # Assume latency range: 500ms (best) to 5000ms (worst)
            latency = max(500, min(5000, metrics.avg_latency_ms))
            speed_score = 1 - ((latency - 500) / 4500)

            # Cost score (simple heuristic, 0-1, higher is better = cheaper)
            cost_scores = {
                "google": 1.0,
                "anthropic": 0.7,
                "openai": 0.5,
            }
            cost_score = cost_scores.get(provider.name, 0.5)

            # Weighted total
            total_score = (
                success_score * 0.4 +
                speed_score * 0.3 +
                cost_score * 0.3
            )

            scores.append((provider, total_score))

        # Return provider with highest score
        return max(scores, key=lambda x: x[1])[0]

    def _select_round_robin(
        self,
        providers: List[BaseProvider],
    ) -> BaseProvider:
        """
        Select provider using round-robin rotation.

        Distributes load evenly across all providers.

        Args:
            providers: Available providers

        Returns:
            Next provider in rotation
        """
        if not providers:
            return None

        # Get next provider
        provider = providers[self._round_robin_index % len(providers)]

        # Increment for next time
        self._round_robin_index += 1

        return provider

    # ══════════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ══════════════════════════════════════════════════════════════════════

    def get_all_providers(self) -> List[BaseProvider]:
        """Get list of all providers"""
        return self._providers.copy()

    def get_healthy_providers(self) -> List[BaseProvider]:
        """Get list of currently healthy providers"""
        return [p for p in self._providers if p.is_healthy()]

    def get_provider_by_name(self, name: str) -> Optional[BaseProvider]:
        """
        Get specific provider by name.

        Args:
            name: Provider name (e.g., "anthropic")

        Returns:
            Provider or None if not found
        """
        for provider in self._providers:
            if provider.name == name:
                return provider
        return None

    def set_strategy(self, strategy: RoutingStrategy):
        """
        Change default routing strategy.

        Args:
            strategy: New strategy to use
        """
        self._strategy = strategy

    def set_fallback_strategy(self, strategy: str):
        """Change fallback ordering strategy."""
        self._fallback_strategy = strategy

    def add_provider(self, provider: BaseProvider):
        """
        Add a new provider to the router.

        Args:
            provider: Provider to add
        """
        if provider not in self._providers:
            self._providers.append(provider)

    def remove_provider(self, provider_name: str):
        """
        Remove a provider from the router.

        Args:
            provider_name: Name of provider to remove
        """
        self._providers = [
            p for p in self._providers
            if p.name != provider_name
        ]
