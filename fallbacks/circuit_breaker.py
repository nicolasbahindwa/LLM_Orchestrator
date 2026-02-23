"""
Circuit Breaker Pattern - Prevent cascading failures.

Implements the circuit breaker pattern to protect against cascading failures
by temporarily disabling providers that are experiencing issues.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Too many failures, requests are rejected
- HALF_OPEN: Testing if service has recovered

Prevents:
- Wasting time on broken services
- Cascading failures across the system
- Resource exhaustion from retrying failed services
"""

import time
import logging
import inspect
from typing import Optional, Callable, Any, Dict
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
from functools import wraps

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()      # Normal operation - requests flow through
    OPEN = auto()        # Circuit is open - rejecting requests
    HALF_OPEN = auto()   # Testing recovery - limited requests allowed


@dataclass
class CircuitBreakerConfig:
    """
    Circuit breaker configuration.

    Attributes:
        failure_threshold: Number of failures before opening circuit
        success_threshold: Number of successes in half-open before closing
        recovery_timeout_seconds: Time to wait before attempting recovery
        half_open_max_requests: Max requests allowed in half-open state
        expected_errors: Error types that count toward failure threshold
    """
    failure_threshold: int = 5
    success_threshold: int = 2
    recovery_timeout_seconds: int = 60
    half_open_max_requests: int = 3
    expected_errors: tuple = field(default_factory=lambda: (Exception,))


@dataclass
class CircuitBreakerMetrics:
    """Metrics tracked by circuit breaker."""
    state_changes: int = 0
    total_failures: int = 0
    total_successes: int = 0
    rejected_requests: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    opened_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    half_open_attempts: int = 0


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Circuit breaker implementation.

    Protects services from cascading failures by tracking errors
    and temporarily disabling failing services.

    Example:
        breaker = CircuitBreaker(
            name="anthropic_provider",
            config=CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout_seconds=60
            )
        )

        @breaker
        async def call_api():
            # Your API call
            pass

        # Or use context manager
        with breaker:
            result = await call_api()
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Name of the circuit breaker (e.g., provider name)
            config: Configuration (uses defaults if not provided)
            on_state_change: Callback function for state changes
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._on_state_change = on_state_change

        # State
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_requests = 0

        # Metrics
        self.metrics = CircuitBreakerMetrics()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting requests)."""
        return self._state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self._state == CircuitState.HALF_OPEN

    def __call__(self, func):
        """
        Decorator to wrap functions with circuit breaker.

        Usage:
            @breaker
            async def my_function():
                pass
        """
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Check if we can proceed
            self._before_call()

            try:
                # Execute function
                result = await func(*args, **kwargs)

                # Record success
                self._on_success()

                return result

            except Exception as e:
                # Record failure
                self._on_failure(e)
                raise

        return wrapper

    async def execute(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute an async/sync callable under circuit breaker protection.

        This is the preferred integration point for orchestrator/provider calls.
        """
        self._before_call()
        try:
            result = func(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise

    def __enter__(self):
        """Context manager entry."""
        self._before_call()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is None:
            self._on_success()
        elif isinstance(exc_val, self.config.expected_errors):
            self._on_failure(exc_val)
        # Propagate exception
        return False

    def _before_call(self):
        """Check circuit state before allowing call."""
        if self._state == CircuitState.OPEN:
            # Check if we should attempt recovery
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                self.metrics.rejected_requests += 1
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Rejecting request. Will retry after recovery timeout."
                )

        elif self._state == CircuitState.HALF_OPEN:
            # Limit requests in half-open state
            if self._half_open_requests >= self.config.half_open_max_requests:
                self.metrics.rejected_requests += 1
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is HALF_OPEN. "
                    f"Maximum test requests ({self.config.half_open_max_requests}) reached."
                )
            self._half_open_requests += 1
            self.metrics.half_open_attempts += 1

    def _on_success(self):
        """Handle successful call."""
        self._success_count += 1
        self.metrics.total_successes += 1
        self.metrics.last_success_time = datetime.now()

        if self._state == CircuitState.HALF_OPEN:
            # Check if we've had enough successes to close circuit
            if self._success_count >= self.config.success_threshold:
                self._transition_to_closed()

    def _on_failure(self, exception: Exception):
        """Handle failed call."""
        # Only count expected errors
        if not isinstance(exception, self.config.expected_errors):
            return

        self._failure_count += 1
        self._last_failure_time = time.time()
        self.metrics.total_failures += 1
        self.metrics.last_failure_time = datetime.now()

        logger.warning(
            f"Circuit breaker '{self.name}': Failure recorded. "
            f"Count: {self._failure_count}/{self.config.failure_threshold}"
        )

        # Check if we should open circuit
        if self._state == CircuitState.CLOSED:
            if self._failure_count >= self.config.failure_threshold:
                self._transition_to_open()

        elif self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open immediately reopens circuit
            self._transition_to_open()

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self._last_failure_time is None:
            return True

        elapsed = time.time() - self._last_failure_time
        return elapsed >= self.config.recovery_timeout_seconds

    def _transition_to_open(self):
        """Transition to OPEN state."""
        old_state = self._state
        self._state = CircuitState.OPEN
        self.metrics.opened_at = datetime.now()
        self.metrics.state_changes += 1

        logger.error(
            f"Circuit breaker '{self.name}': OPENED. "
            f"Threshold reached: {self._failure_count} failures. "
            f"Will retry after {self.config.recovery_timeout_seconds}s"
        )

        if self._on_state_change:
            self._on_state_change(old_state, self._state)

    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        old_state = self._state
        self._state = CircuitState.HALF_OPEN
        self._half_open_requests = 0
        self._success_count = 0
        self._failure_count = 0
        self.metrics.state_changes += 1

        logger.info(
            f"Circuit breaker '{self.name}': HALF_OPEN. "
            f"Testing recovery with up to {self.config.half_open_max_requests} requests"
        )

        if self._on_state_change:
            self._on_state_change(old_state, self._state)

    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        old_state = self._state
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_requests = 0
        self.metrics.closed_at = datetime.now()
        self.metrics.state_changes += 1

        logger.info(
            f"Circuit breaker '{self.name}': CLOSED. "
            f"Service recovered successfully"
        )

        if self._on_state_change:
            self._on_state_change(old_state, self._state)

    def reset(self):
        """Manually reset circuit breaker to CLOSED state."""
        logger.info(f"Circuit breaker '{self.name}': Manual reset")
        self._transition_to_closed()

    def get_status(self) -> Dict[str, Any]:
        """
        Get current status and metrics.

        Returns:
            Dictionary with state and metrics
        """
        return {
            "name": self.name,
            "state": self._state.name,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "metrics": {
                "state_changes": self.metrics.state_changes,
                "total_failures": self.metrics.total_failures,
                "total_successes": self.metrics.total_successes,
                "rejected_requests": self.metrics.rejected_requests,
                "last_failure": self.metrics.last_failure_time.isoformat() if self.metrics.last_failure_time else None,
                "last_success": self.metrics.last_success_time.isoformat() if self.metrics.last_success_time else None,
            }
        }


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.

    Centralized management of circuit breakers for all providers.

    Example:
        registry = CircuitBreakerRegistry()

        # Get or create breaker for provider
        breaker = registry.get_breaker("anthropic")

        # Get status of all breakers
        status = registry.get_all_status()
    """

    def __init__(self):
        """Initialize circuit breaker registry."""
        self._breakers: Dict[str, CircuitBreaker] = {}

    def get_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """
        Get or create circuit breaker for a service.

        Args:
            name: Service name
            config: Configuration (uses defaults if not provided)

        Returns:
            CircuitBreaker instance
        """
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name=name, config=config)

        return self._breakers[name]

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all circuit breakers.

        Returns:
            Dictionary mapping names to status
        """
        return {
            name: breaker.get_status()
            for name, breaker in self._breakers.items()
        }

    def reset_all(self):
        """Reset all circuit breakers to CLOSED state."""
        for breaker in self._breakers.values():
            breaker.reset()

    def get_healthy_services(self) -> list[str]:
        """
        Get list of services with closed circuit breakers.

        Returns:
            List of healthy service names
        """
        return [
            name for name, breaker in self._breakers.items()
            if breaker.is_closed
        ]

    def get_unhealthy_services(self) -> list[str]:
        """
        Get list of services with open circuit breakers.

        Returns:
            List of unhealthy service names
        """
        return [
            name for name, breaker in self._breakers.items()
            if breaker.is_open
        ]
