"""
Metrics Collector - Comprehensive performance tracking.

Collects and aggregates metrics for observability and optimization:
- Latency percentiles (p50, p95, p99)
- Token usage and costs
- Success/error rates
- Provider performance
- Cache hit rates

Critical for:
- Performance monitoring
- Cost optimization
- SLA tracking
- Capacity planning
"""

import time
import statistics
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque


@dataclass
class RequestMetrics:
    """
    Detailed metrics for a single request.

    Tracks everything needed for observability and cost optimization.
    """
    # Identity
    request_id: str
    provider: str
    model: str

    # Timing
    start_time: datetime
    end_time: Optional[datetime] = None
    latency_ms: float = 0.0
    time_to_first_token_ms: Optional[float] = None

    # Tokens
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Cost
    cost_usd: float = 0.0

    # Status
    success: bool = False
    error: Optional[str] = None
    cached: bool = False
    fallback_used: bool = False
    retry_count: int = 0

    # Routing
    routing_strategy: str = ""
    task_complexity: str = ""

    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)

    def finalize(self):
        """Finalize metrics after request completion."""
        if self.end_time and self.start_time:
            self.latency_ms = (self.end_time - self.start_time).total_seconds() * 1000


class MetricsCollector:
    """
    Comprehensive metrics collector for LLM orchestration.

    Tracks and aggregates metrics for:
    - Performance monitoring (latency percentiles)
    - Cost tracking (per provider/model)
    - Reliability (success rates)
    - Efficiency (cache hit rates, token usage)

    Example:
        collector = MetricsCollector(max_history=10000)

        # Record request
        metrics = RequestMetrics(
            request_id="req_123",
            provider="anthropic",
            model="claude-sonnet-4",
            start_time=datetime.now(),
            ...
        )
        collector.record_request(metrics)

        # Get statistics
        stats = collector.get_stats()
        print(f"P95 latency: {stats['latency_p95']}ms")
        print(f"Total cost: ${stats['total_cost']}")
    """

    def __init__(self, max_history: int = 10000):
        """
        Initialize metrics collector.

        Args:
            max_history: Maximum number of requests to keep in history
                        (uses sliding window for percentile calculations)
        """
        self.max_history = max_history

        # Request history (for percentile calculations)
        self._metrics: deque[RequestMetrics] = deque(maxlen=max_history)

        # Aggregated metrics by provider
        self._provider_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "requests": 0,
                "successes": 0,
                "failures": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "latencies": deque(maxlen=1000),  # Last 1000 for percentiles
            }
        )

        # Aggregated metrics by model
        self._model_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "requests": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
            }
        )

        # Global counters
        self._total_requests = 0
        self._total_successes = 0
        self._total_failures = 0
        self._total_cache_hits = 0
        self._total_cache_misses = 0
        self._total_fallback_uses = 0
        self._total_cost = 0.0
        self._total_tokens = 0

    def record_request(self, metrics: RequestMetrics):
        """
        Record metrics for a completed request.

        Args:
            metrics: Request metrics to record
        """
        # Finalize timing if not already done
        if not metrics.latency_ms and metrics.end_time:
            metrics.finalize()

        # Add to history
        self._metrics.append(metrics)

        # Update global counters
        self._total_requests += 1
        if metrics.success:
            self._total_successes += 1
        else:
            self._total_failures += 1

        if metrics.cached:
            self._total_cache_hits += 1
        else:
            self._total_cache_misses += 1

        if metrics.fallback_used:
            self._total_fallback_uses += 1

        self._total_cost += metrics.cost_usd
        self._total_tokens += metrics.total_tokens

        # Update provider stats
        provider_stats = self._provider_stats[metrics.provider]
        provider_stats["requests"] += 1
        if metrics.success:
            provider_stats["successes"] += 1
        else:
            provider_stats["failures"] += 1
        provider_stats["total_tokens"] += metrics.total_tokens
        provider_stats["total_cost"] += metrics.cost_usd
        if metrics.latency_ms > 0:
            provider_stats["latencies"].append(metrics.latency_ms)

        # Update model stats
        model_stats = self._model_stats[metrics.model]
        model_stats["requests"] += 1
        model_stats["total_tokens"] += metrics.total_tokens
        model_stats["total_cost"] += metrics.cost_usd

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics.

        Returns:
            Dictionary with all metrics and aggregations

        Example:
            stats = collector.get_stats()
            {
                "total_requests": 1000,
                "success_rate_percent": 98.5,
                "latency_p50": 1200.5,
                "latency_p95": 3500.2,
                "latency_p99": 5000.0,
                "total_cost": 1.25,
                "cache_hit_rate_percent": 45.0,
                "providers": {...},
                "models": {...}
            }
        """
        # Calculate success rate
        success_rate = (
            (self._total_successes / self._total_requests * 100)
            if self._total_requests > 0 else 0.0
        )

        # Calculate cache hit rate
        cache_hit_rate = (
            (self._total_cache_hits / self._total_requests * 100)
            if self._total_requests > 0 else 0.0
        )

        # Calculate latency percentiles (all successful requests)
        all_latencies = [
            m.latency_ms for m in self._metrics
            if m.success and m.latency_ms > 0
        ]

        latency_stats = {}
        if all_latencies:
            latency_stats = {
                "latency_p50": round(statistics.median(all_latencies), 2),
                "latency_p95": round(self._percentile(all_latencies, 0.95), 2),
                "latency_p99": round(self._percentile(all_latencies, 0.99), 2),
                "latency_avg": round(statistics.mean(all_latencies), 2),
                "latency_min": round(min(all_latencies), 2),
                "latency_max": round(max(all_latencies), 2),
            }

        # Build comprehensive stats
        stats = {
            # Global metrics
            "total_requests": self._total_requests,
            "successful_requests": self._total_successes,
            "failed_requests": self._total_failures,
            "success_rate_percent": round(success_rate, 2),

            # Latency
            **latency_stats,

            # Cost and tokens
            "total_cost_usd": round(self._total_cost, 4),
            "total_tokens": self._total_tokens,
            "avg_cost_per_request": (
                round(self._total_cost / self._total_requests, 4)
                if self._total_requests > 0 else 0.0
            ),
            "avg_tokens_per_request": (
                round(self._total_tokens / self._total_requests, 2)
                if self._total_requests > 0 else 0.0
            ),

            # Cache
            "cache_hits": self._total_cache_hits,
            "cache_misses": self._total_cache_misses,
            "cache_hit_rate_percent": round(cache_hit_rate, 2),

            # Fallbacks
            "fallback_uses": self._total_fallback_uses,
            "fallback_rate_percent": (
                round(self._total_fallback_uses / self._total_requests * 100, 2)
                if self._total_requests > 0 else 0.0
            ),

            # Per-provider breakdown
            "providers": self._get_provider_stats(),

            # Per-model breakdown
            "models": self._get_model_stats(),
        }

        return stats

    def _get_provider_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get per-provider statistics."""
        provider_stats = {}

        for provider, stats in self._provider_stats.items():
            success_rate = (
                (stats["successes"] / stats["requests"] * 100)
                if stats["requests"] > 0 else 0.0
            )

            # Calculate latency percentiles for this provider
            latencies = list(stats["latencies"])
            latency_stats = {}
            if latencies:
                latency_stats = {
                    "latency_p50": round(statistics.median(latencies), 2),
                    "latency_p95": round(self._percentile(latencies, 0.95), 2),
                    "latency_avg": round(statistics.mean(latencies), 2),
                }

            provider_stats[provider] = {
                "requests": stats["requests"],
                "successes": stats["successes"],
                "failures": stats["failures"],
                "success_rate_percent": round(success_rate, 2),
                "total_tokens": stats["total_tokens"],
                "total_cost_usd": round(stats["total_cost"], 4),
                **latency_stats,
            }

        return provider_stats

    def _get_model_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get per-model statistics."""
        model_stats = {}

        for model, stats in self._model_stats.items():
            model_stats[model] = {
                "requests": stats["requests"],
                "total_tokens": stats["total_tokens"],
                "total_cost_usd": round(stats["total_cost"], 4),
                "avg_cost_per_request": (
                    round(stats["total_cost"] / stats["requests"], 4)
                    if stats["requests"] > 0 else 0.0
                ),
            }

        return model_stats

    def _percentile(self, data: List[float], percentile: float) -> float:
        """
        Calculate percentile of data.

        Args:
            data: List of numbers
            percentile: Percentile to calculate (0.0-1.0)

        Returns:
            Value at given percentile
        """
        if not data:
            return 0.0

        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        index = min(index, len(sorted_data) - 1)

        return sorted_data[index]

    def reset(self):
        """Reset all metrics to zero."""
        self._metrics.clear()
        self._provider_stats.clear()
        self._model_stats.clear()
        self._total_requests = 0
        self._total_successes = 0
        self._total_failures = 0
        self._total_cache_hits = 0
        self._total_cache_misses = 0
        self._total_fallback_uses = 0
        self._total_cost = 0.0
        self._total_tokens = 0

    def get_recent_requests(self, count: int = 10) -> List[RequestMetrics]:
        """
        Get most recent requests.

        Args:
            count: Number of recent requests to return

        Returns:
            List of recent RequestMetrics
        """
        return list(self._metrics)[-count:]

    def export_metrics(self) -> List[Dict[str, Any]]:
        """
        Export all metrics as list of dictionaries.

        Useful for sending to external monitoring systems.

        Returns:
            List of metric dictionaries
        """
        return [
            {
                "request_id": m.request_id,
                "provider": m.provider,
                "model": m.model,
                "latency_ms": m.latency_ms,
                "input_tokens": m.input_tokens,
                "output_tokens": m.output_tokens,
                "cost_usd": m.cost_usd,
                "success": m.success,
                "cached": m.cached,
                "fallback_used": m.fallback_used,
                "timestamp": m.start_time.isoformat(),
            }
            for m in self._metrics
        ]
