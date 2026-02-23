"""
Config schema for standalone LLM orchestrator.

This module defines the configuration structure that can be loaded from:
- YAML/JSON files
- Python dictionaries
- Environment variables
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


# ══════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════

class ComplexityLevel(str, Enum):
    """Model complexity level"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class CostTier(str, Enum):
    """Cost tier classification"""
    FREE = "free"          # $0 (Ollama)
    CHEAP = "cheap"        # < $1/million output
    BALANCED = "balanced"  # $1-$20/million output
    EXPENSIVE = "expensive"  # > $20/million output


class SpeedTier(str, Enum):
    """Response speed tier"""
    VERY_FAST = "very_fast"  # < 1s
    FAST = "fast"            # 1-2s
    MEDIUM = "medium"        # 2-5s
    SLOW = "slow"            # > 5s


class QualityTier(str, Enum):
    """Model quality tier"""
    GOOD = "good"
    EXCELLENT = "excellent"
    BEST = "best"


class RoutingStrategy(str, Enum):
    """Routing strategy"""
    COST = "cost"           # Cheapest model
    QUALITY = "quality"     # Best quality model
    SPEED = "speed"         # Fastest model
    BALANCED = "balanced"   # Optimal mix
    ROUND_ROBIN = "round_robin"  # Equal distribution


class RetryStrategy(str, Enum):
    """Retry backoff strategy"""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    CONSTANT = "constant"
    FIBONACCI = "fibonacci"


# ══════════════════════════════════════════════════════════════════════
# Model Configuration
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    """
    Configuration for a single model.

    This defines model capabilities and characteristics
    for intelligent routing.
    """
    name: str
    provider: str
    complexity_level: ComplexityLevel
    cost_tier: CostTier
    speed_tier: SpeedTier
    quality_tier: QualityTier
    max_tokens: int = 4096
    context_window: int = 8192

    # Pricing (optional - can be in separate pricing config)
    input_price_per_million: Optional[float] = None
    output_price_per_million: Optional[float] = None


# ══════════════════════════════════════════════════════════════════════
# Provider Configuration
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ProviderConfig:
    """Configuration for a provider"""
    enabled: bool = True
    adapter: Optional[str] = None  # Adapter type: anthropic, openai_compatible, ollama
    adapter_config: Dict[str, Any] = field(default_factory=dict)
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 60
    default_model: Optional[str] = None
    available_models: List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════
# Routing Configuration
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ComplexityRoutingConfig:
    """Complexity-based routing configuration"""
    enabled: bool = True

    # Map task complexity → allowed model complexity levels
    # e.g., SIMPLE task can use SIMPLE models
    # COMPLEX task can use MODERATE or COMPLEX models
    simple_allows: List[ComplexityLevel] = field(
        default_factory=lambda: [ComplexityLevel.SIMPLE]
    )
    moderate_allows: List[ComplexityLevel] = field(
        default_factory=lambda: [ComplexityLevel.SIMPLE, ComplexityLevel.MODERATE]
    )
    complex_allows: List[ComplexityLevel] = field(
        default_factory=lambda: [ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX]
    )
    very_complex_allows: List[ComplexityLevel] = field(
        default_factory=lambda: [ComplexityLevel.COMPLEX, ComplexityLevel.VERY_COMPLEX]
    )


@dataclass
class RoutingConfig:
    """Routing configuration"""
    strategy: RoutingStrategy = RoutingStrategy.BALANCED
    complexity_routing: ComplexityRoutingConfig = field(default_factory=ComplexityRoutingConfig)
    fallback_strategy: str = "same_provider_first"  # or "cheapest_first", "fastest_first"
    max_fallback_attempts: int = 3


# ══════════════════════════════════════════════════════════════════════
# Cache Configuration
# ══════════════════════════════════════════════════════════════════════

@dataclass
class CacheConfig:
    """Cache configuration"""
    enabled: bool = True
    backend: str = "memory"  # memory, redis
    ttl: int = 3600
    max_size: int = 1000

    # Redis-specific settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_username: Optional[str] = None
    redis_prefix: str = "llm_cache:"
    redis_max_connections: int = 50
    redis_socket_timeout: int = 5


# ══════════════════════════════════════════════════════════════════════
# Resilience Configuration
# ══════════════════════════════════════════════════════════════════════

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout: int = 60


@dataclass
class RetryConfig:
    """Retry configuration"""
    enabled: bool = True
    max_retries: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    base_delay: float = 1.0
    max_delay: float = 60.0


@dataclass
class ResilienceConfig:
    """Resilience configuration"""
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)


# ══════════════════════════════════════════════════════════════════════
# Monitoring Configuration
# ══════════════════════════════════════════════════════════════════════

@dataclass
class LoggingConfig:
    """Logging configuration"""
    enabled: bool = True
    level: str = "INFO"
    format: str = "json"  # json, text
    output: str = "stdout"  # stdout, file, both
    file_path: Optional[str] = None


@dataclass
class MetricsConfig:
    """Metrics configuration"""
    enabled: bool = True
    track_costs: bool = True
    track_latency_percentiles: bool = True
    prometheus_enabled: bool = False
    prometheus_host: str = "0.0.0.0"
    prometheus_port: int = 9464
    prometheus_start_http_server: bool = True
    otel_enabled: bool = False
    otel_service_name: str = "llm-orchestrator"
    otel_exporter_endpoint: Optional[str] = None
    otel_exporter_insecure: bool = True
    sinks: List[str] = field(default_factory=list)


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)


# ══════════════════════════════════════════════════════════════════════
# Main Orchestrator Configuration
# ══════════════════════════════════════════════════════════════════════

@dataclass
class OrchestratorConfig:
    """
    Main orchestrator configuration.

    This is the top-level config that contains all settings.
    Can be loaded from YAML, JSON, dict, or environment variables.
    """
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    resilience: ResilienceConfig = field(default_factory=ResilienceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    def get_enabled_models(self) -> List[ModelConfig]:
        """Get list of models from enabled providers"""
        enabled_models = []

        for model_name, model_config in self.models.items():
            provider_config = self.providers.get(model_config.provider)
            if provider_config and provider_config.enabled:
                # Check if model is in provider's available_models (if specified)
                if not provider_config.available_models or model_name in provider_config.available_models:
                    enabled_models.append(model_config)

        return enabled_models

    def get_models_for_complexity(self, task_complexity: ComplexityLevel) -> List[ModelConfig]:
        """
        Get models that can handle a given task complexity.

        This is the KEY routing logic!
        """
        # Get allowed model complexity levels for this task
        if task_complexity == ComplexityLevel.SIMPLE:
            allowed = self.routing.complexity_routing.simple_allows
        elif task_complexity == ComplexityLevel.MODERATE:
            allowed = self.routing.complexity_routing.moderate_allows
        elif task_complexity == ComplexityLevel.COMPLEX:
            allowed = self.routing.complexity_routing.complex_allows
        else:  # VERY_COMPLEX
            allowed = self.routing.complexity_routing.very_complex_allows

        # Filter models by allowed complexity levels
        enabled_models = self.get_enabled_models()
        return [m for m in enabled_models if m.complexity_level in allowed]

    def get_models_by_provider(self, provider_name: str) -> List[ModelConfig]:
        """Get all models for a specific provider"""
        enabled_models = self.get_enabled_models()
        return [m for m in enabled_models if m.provider == provider_name]

    def get_cheapest_model(self, models: List[ModelConfig]) -> Optional[ModelConfig]:
        """Get cheapest model from list"""
        free_models = [m for m in models if m.cost_tier == CostTier.FREE]
        if free_models:
            return free_models[0]  # All FREE models have same cost

        cheap_models = [m for m in models if m.cost_tier == CostTier.CHEAP]
        if cheap_models:
            # Sort by actual price if available
            if cheap_models[0].output_price_per_million is not None:
                return min(cheap_models, key=lambda m: m.output_price_per_million or 0)
            return cheap_models[0]

        # No free or cheap, return any
        if models:
            return models[0]

        return None
