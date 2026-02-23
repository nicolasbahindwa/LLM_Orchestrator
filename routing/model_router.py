"""
Model Router - Intelligent MODEL selection (not just provider).

This is an ENHANCED router that routes to SPECIFIC MODELS based on:
- Task complexity (simple → Haiku, complex → Opus)
- Cost tier (FREE → Ollama, CHEAP → Haiku/GPT-4o-mini)
- Speed requirements
- Quality requirements

KEY IMPROVEMENT:
Old router: "Use Anthropic vs OpenAI"
New router: "Use claude-haiku-4 vs claude-sonnet-4 vs gpt-4o-mini"
"""

from typing import List, Optional, Dict, Any
from ..config import (
    OrchestratorConfig,
    ModelConfig,
    ComplexityLevel,
    CostTier,
    SpeedTier,
    QualityTier,
    RoutingStrategy,
)


class ModelRouter:
    """
    Enhanced router that selects SPECIFIC MODELS, not just providers.

    This enables:
    - Routing simple tasks to cheap models (Haiku) even within same provider
    - Routing complex tasks to powerful models (Opus) even if more expensive
    - Using FREE models (Ollama) when cost is priority
    - Falling back within same provider (Haiku → Sonnet → Opus)

    Example:
        router = ModelRouter(config)

        # Simple task
        model = router.select_model(
            task_complexity=ComplexityLevel.SIMPLE,
            strategy=RoutingStrategy.COST
        )
        # → Returns claude-haiku-4 or llama3.2 (FREE!)

        # Complex task
        model = router.select_model(
            task_complexity=ComplexityLevel.VERY_COMPLEX,
            strategy=RoutingStrategy.QUALITY
        )
        # → Returns claude-opus-4 or gpt-4o
    """

    def __init__(self, config: OrchestratorConfig):
        """
        Initialize model router.

        Args:
            config: Orchestrator configuration with models catalog
        """
        self.config = config
        self._round_robin_index = 0

    def select_model(
        self,
        task_complexity: ComplexityLevel,
        strategy: Optional[RoutingStrategy] = None,
        provider_filter: Optional[str] = None,
    ) -> Optional[ModelConfig]:
        """
        Select best model for a task.

        Process:
        1. Filter models by task complexity (simple → simple models)
        2. Optionally filter by provider
        3. Apply routing strategy (cost/quality/speed)
        4. Return best model

        Args:
            task_complexity: Task complexity level
            strategy: Routing strategy (uses config default if None)
            provider_filter: Only consider this provider (optional)

        Returns:
            Selected model or None if no models available

        Example:
            # Simple task, minimize cost
            model = router.select_model(
                task_complexity=ComplexityLevel.SIMPLE,
                strategy=RoutingStrategy.COST
            )
            # → claude-haiku-4 or llama3.2 (FREE)

            # Complex task, only Anthropic
            model = router.select_model(
                task_complexity=ComplexityLevel.COMPLEX,
                strategy=RoutingStrategy.QUALITY,
                provider_filter="anthropic"
            )
            # → claude-opus-4
        """
        # Use config default strategy if not provided
        if strategy is None:
            strategy = self.config.routing.strategy

        # Step 1: Filter by complexity
        candidate_models = self.config.get_models_for_complexity(task_complexity)

        if not candidate_models:
            return None

        # Step 2: Filter by provider if specified
        if provider_filter:
            candidate_models = [
                m for m in candidate_models
                if m.provider == provider_filter
            ]

        if not candidate_models:
            return None

        # Step 3: Apply routing strategy
        if strategy == RoutingStrategy.COST:
            return self._select_cheapest(candidate_models)

        elif strategy == RoutingStrategy.QUALITY:
            return self._select_highest_quality(candidate_models)

        elif strategy == RoutingStrategy.SPEED:
            return self._select_fastest(candidate_models)

        elif strategy == RoutingStrategy.BALANCED:
            return self._select_balanced(candidate_models)

        elif strategy == RoutingStrategy.ROUND_ROBIN:
            return self._select_round_robin(candidate_models)

        else:
            # Fallback to first model
            return candidate_models[0]

    def get_fallback_models(
        self,
        primary_model: ModelConfig,
        task_complexity: ComplexityLevel,
        strategy: str = "same_provider_first",
    ) -> List[ModelConfig]:
        """
        Get ordered list of fallback models.

        Fallback strategies:
        - same_provider_first: Try other models from same provider first
        - upgrade_complexity: Try more powerful model from same provider
        - cheapest_first: Try cheapest available alternatives
        - fastest_first: Try fastest available alternatives

        Args:
            primary_model: The model that failed
            task_complexity: Task complexity level
            strategy: Fallback strategy

        Returns:
            Ordered list of fallback models

        Example:
            # Primary: claude-sonnet-4 failed
            fallbacks = router.get_fallback_models(
                primary_model=sonnet,
                task_complexity=ComplexityLevel.COMPLEX,
                strategy="same_provider_first"
            )
            # → [claude-opus-4, gpt-4o, mixtral, ...]
        """
        # Get all models that can handle this complexity
        all_models = self.config.get_models_for_complexity(task_complexity)

        # Exclude the primary model
        fallbacks = [m for m in all_models if m.name != primary_model.name]

        if strategy == "same_provider_first":
            # Prioritize models from same provider
            same_provider = [m for m in fallbacks if m.provider == primary_model.provider]
            other_providers = [m for m in fallbacks if m.provider != primary_model.provider]

            # Sort same-provider models by complexity (higher complexity first)
            complexity_order = {
                ComplexityLevel.SIMPLE: 1,
                ComplexityLevel.MODERATE: 2,
                ComplexityLevel.COMPLEX: 3,
                ComplexityLevel.VERY_COMPLEX: 4,
            }
            same_provider.sort(
                key=lambda m: complexity_order.get(m.complexity_level, 0),
                reverse=True
            )

            # Sort other providers by success/quality
            other_providers.sort(
                key=lambda m: (m.quality_tier.value, -self._get_cost_score(m))
            )

            return same_provider + other_providers

        elif strategy == "upgrade_complexity":
            # Try more powerful models from same provider first
            same_provider = [m for m in fallbacks if m.provider == primary_model.provider]

            complexity_order = {
                ComplexityLevel.SIMPLE: 1,
                ComplexityLevel.MODERATE: 2,
                ComplexityLevel.COMPLEX: 3,
                ComplexityLevel.VERY_COMPLEX: 4,
            }
            primary_complexity = complexity_order.get(primary_model.complexity_level, 2)

            # Only include models with higher complexity
            upgrades = [
                m for m in same_provider
                if complexity_order.get(m.complexity_level, 0) > primary_complexity
            ]

            # Then add all other options
            others = [m for m in fallbacks if m not in upgrades]

            return upgrades + others

        elif strategy == "cheapest_first":
            # Sort by cost tier (FREE → CHEAP → BALANCED → EXPENSIVE)
            return sorted(fallbacks, key=self._get_cost_score, reverse=True)

        elif strategy == "fastest_first":
            # Sort by speed tier
            speed_order = {
                SpeedTier.VERY_FAST: 4,
                SpeedTier.FAST: 3,
                SpeedTier.MEDIUM: 2,
                SpeedTier.SLOW: 1,
            }
            return sorted(
                fallbacks,
                key=lambda m: speed_order.get(m.speed_tier, 2),
                reverse=True
            )

        else:
            # Default: same as same_provider_first
            return self.get_fallback_models(
                primary_model,
                task_complexity,
                "same_provider_first"
            )

    # ══════════════════════════════════════════════════════════════════════
    # ROUTING STRATEGY IMPLEMENTATIONS
    # ══════════════════════════════════════════════════════════════════════

    def _select_cheapest(self, models: List[ModelConfig]) -> ModelConfig:
        """
        Select cheapest model.

        Priority:
        1. FREE models (Ollama)
        2. CHEAP models sorted by actual price
        3. Any model sorted by cost_tier

        Args:
            models: Candidate models

        Returns:
            Cheapest model
        """
        # Prioritize FREE models
        free_models = [m for m in models if m.cost_tier == CostTier.FREE]
        if free_models:
            return free_models[0]  # All FREE models cost same ($0)

        # Sort by cost tier and actual price
        return min(models, key=lambda m: (
            self._get_cost_tier_value(m.cost_tier),
            m.output_price_per_million or 999.0
        ))

    def _select_highest_quality(self, models: List[ModelConfig]) -> ModelConfig:
        """
        Select highest quality model.

        Priority:
        1. BEST quality tier
        2. Higher complexity level (more capable)
        3. Lowest cost within same quality

        Args:
            models: Candidate models

        Returns:
            Highest quality model
        """
        quality_order = {
            QualityTier.GOOD: 1,
            QualityTier.EXCELLENT: 2,
            QualityTier.BEST: 3,
        }

        complexity_order = {
            ComplexityLevel.SIMPLE: 1,
            ComplexityLevel.MODERATE: 2,
            ComplexityLevel.COMPLEX: 3,
            ComplexityLevel.VERY_COMPLEX: 4,
        }

        return max(models, key=lambda m: (
            quality_order.get(m.quality_tier, 1),
            complexity_order.get(m.complexity_level, 2)
        ))

    def _select_fastest(self, models: List[ModelConfig]) -> ModelConfig:
        """
        Select fastest model.

        Priority:
        1. VERY_FAST tier
        2. FAST tier
        3. Others

        Args:
            models: Candidate models

        Returns:
            Fastest model
        """
        speed_order = {
            SpeedTier.VERY_FAST: 4,
            SpeedTier.FAST: 3,
            SpeedTier.MEDIUM: 2,
            SpeedTier.SLOW: 1,
        }

        return max(models, key=lambda m: speed_order.get(m.speed_tier, 2))

    def _select_balanced(self, models: List[ModelConfig]) -> ModelConfig:
        """
        Select best balanced model.

        Scoring formula:
        score = (quality_score * 0.35) + (speed_score * 0.35) + (cost_score * 0.30)

        Args:
            models: Candidate models

        Returns:
            Best balanced model
        """
        def score_model(model: ModelConfig) -> float:
            # Quality score (0-1)
            quality_scores = {
                QualityTier.GOOD: 0.6,
                QualityTier.EXCELLENT: 0.8,
                QualityTier.BEST: 1.0,
            }
            quality_score = quality_scores.get(model.quality_tier, 0.7)

            # Speed score (0-1)
            speed_scores = {
                SpeedTier.VERY_FAST: 1.0,
                SpeedTier.FAST: 0.8,
                SpeedTier.MEDIUM: 0.5,
                SpeedTier.SLOW: 0.3,
            }
            speed_score = speed_scores.get(model.speed_tier, 0.5)

            # Cost score (0-1, higher = cheaper)
            cost_score = self._get_cost_score(model)

            # Weighted total
            return (quality_score * 0.35 + speed_score * 0.35 + cost_score * 0.30)

        return max(models, key=score_model)

    def _select_round_robin(self, models: List[ModelConfig]) -> ModelConfig:
        """
        Select model using round-robin.

        Args:
            models: Candidate models

        Returns:
            Next model in rotation
        """
        if not models:
            return None

        model = models[self._round_robin_index % len(models)]
        self._round_robin_index += 1
        return model

    # ══════════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ══════════════════════════════════════════════════════════════════════

    def _get_cost_tier_value(self, cost_tier: CostTier) -> int:
        """Get numeric value for cost tier (lower = cheaper)"""
        cost_values = {
            CostTier.FREE: 0,
            CostTier.CHEAP: 1,
            CostTier.BALANCED: 2,
            CostTier.EXPENSIVE: 3,
        }
        return cost_values.get(cost_tier, 2)

    def _get_cost_score(self, model: ModelConfig) -> float:
        """
        Get cost score (0-1, higher = cheaper).

        FREE = 1.0
        CHEAP = 0.8
        BALANCED = 0.5
        EXPENSIVE = 0.2
        """
        cost_scores = {
            CostTier.FREE: 1.0,
            CostTier.CHEAP: 0.8,
            CostTier.BALANCED: 0.5,
            CostTier.EXPENSIVE: 0.2,
        }
        return cost_scores.get(model.cost_tier, 0.5)

    def get_all_models(self) -> List[ModelConfig]:
        """Get all enabled models"""
        return self.config.get_enabled_models()

    def get_model_by_name(self, name: str) -> Optional[ModelConfig]:
        """Get model by name"""
        return self.config.models.get(name)

    def get_models_by_provider(self, provider: str) -> List[ModelConfig]:
        """Get all models for a specific provider"""
        return self.config.get_models_by_provider(provider)
