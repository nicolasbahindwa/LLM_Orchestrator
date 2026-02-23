"""
Retry Handler - Exponential backoff retry logic.

Implements sophisticated retry logic with exponential backoff for transient failures.

Features:
- Exponential backoff with jitter
- Configurable retry strategies
- Selective retry (only transient errors)
- Retry budget tracking
- Detailed retry metrics
"""

import asyncio
import logging
import random
import time
from typing import Optional, Callable, Any, Type, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

import httpx

logger = logging.getLogger(__name__)


class RetryStrategy(str, Enum):
    """Retry strategy types."""
    EXPONENTIAL = "exponential"      # 2^n * base_delay
    LINEAR = "linear"                # n * base_delay
    CONSTANT = "constant"            # Always base_delay
    FIBONACCI = "fibonacci"          # Fibonacci sequence * base_delay


@dataclass
class RetryConfig:
    """
    Retry configuration.

    Attributes:
        max_retries: Maximum number of retry attempts
        base_delay_seconds: Base delay between retries
        max_delay_seconds: Maximum delay between retries
        strategy: Retry strategy to use
        exponential_base: Base for exponential backoff (default 2)
        jitter: Add random jitter to prevent thundering herd (0.0-1.0)
        retry_on_exceptions: Exception types that trigger retry
        timeout_seconds: Total timeout for all retries
    """
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    exponential_base: float = 2.0
    jitter: float = 0.1  # 10% jitter
    retry_on_exceptions: Set[Type[Exception]] = field(
        default_factory=lambda: {
            TimeoutError,
            ConnectionError,
            asyncio.TimeoutError,
            httpx.TimeoutException,
            httpx.NetworkError,
        }
    )
    timeout_seconds: Optional[float] = None


@dataclass
class RetryMetrics:
    """Metrics for retry attempts."""
    total_attempts: int = 0
    successful_retries: int = 0
    failed_retries: int = 0
    total_delay_seconds: float = 0.0
    max_attempts_reached: int = 0


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, attempts: int, last_exception: Exception):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(
            f"Retry exhausted after {attempts} attempts. "
            f"Last error: {type(last_exception).__name__}: {last_exception}"
        )


class RetryHandler:
    """
    Retry handler with exponential backoff.

    Handles transient failures by retrying with increasing delays.

    Example:
        handler = RetryHandler(
            config=RetryConfig(
                max_retries=3,
                base_delay_seconds=1.0,
                strategy=RetryStrategy.EXPONENTIAL
            )
        )

        @handler.retry
        async def unstable_function():
            # May fail occasionally
            return await call_api()

        # Or use directly
        result = await handler.execute(unstable_function)
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry handler.

        Args:
            config: Retry configuration (uses defaults if not provided)
        """
        self.config = config or RetryConfig()
        self.metrics = RetryMetrics()

        # Fibonacci sequence cache
        self._fib_cache = [0, 1]

    def retry(self, func):
        """
        Decorator to add retry logic to async functions.

        Usage:
            @handler.retry
            async def my_function():
                pass
        """
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.execute(func, *args, **kwargs)

        return wrapper

    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from successful function call

        Raises:
            RetryExhaustedError: If all retries are exhausted
        """
        attempt = 0
        last_exception = None
        start_time = time.time()

        while attempt <= self.config.max_retries:
            try:
                self.metrics.total_attempts += 1

                # Check total timeout
                if self.config.timeout_seconds:
                    elapsed = time.time() - start_time
                    if elapsed >= self.config.timeout_seconds:
                        logger.warning(
                            f"Retry timeout reached after {elapsed:.2f}s. "
                            f"Attempt {attempt}/{self.config.max_retries}"
                        )
                        break

                # Execute function
                result = await func(*args, **kwargs)

                # Success!
                if attempt > 0:
                    self.metrics.successful_retries += 1
                    logger.info(
                        f"Retry successful after {attempt} attempts"
                    )

                return result

            except Exception as e:
                last_exception = e

                # Check if we should retry this exception
                if not self._should_retry(e):
                    logger.debug(
                        f"Exception {type(e).__name__} not in retry list. "
                        f"Not retrying."
                    )
                    raise

                # Check if we have retries left
                if attempt >= self.config.max_retries:
                    logger.error(
                        f"Max retries ({self.config.max_retries}) reached. "
                        f"Last error: {type(e).__name__}: {e}"
                    )
                    self.metrics.max_attempts_reached += 1
                    break

                # Calculate delay
                delay = self._calculate_delay(attempt)
                self.metrics.total_delay_seconds += delay

                logger.warning(
                    f"Attempt {attempt + 1}/{self.config.max_retries + 1} failed: "
                    f"{type(e).__name__}: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )

                # Wait before retry
                await asyncio.sleep(delay)

                attempt += 1

        # All retries exhausted
        self.metrics.failed_retries += 1
        raise RetryExhaustedError(attempt, last_exception)

    def _should_retry(self, exception: Exception) -> bool:
        """
        Check if exception should trigger retry.

        Args:
            exception: Exception that was raised

        Returns:
            True if should retry, False otherwise
        """
        return any(
            isinstance(exception, exc_type)
            for exc_type in self.config.retry_on_exceptions
        )

    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for retry attempt.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        if self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.base_delay_seconds * (
                self.config.exponential_base ** attempt
            )

        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay_seconds * (attempt + 1)

        elif self.config.strategy == RetryStrategy.CONSTANT:
            delay = self.config.base_delay_seconds

        elif self.config.strategy == RetryStrategy.FIBONACCI:
            fib = self._get_fibonacci(attempt + 1)
            delay = self.config.base_delay_seconds * fib

        else:
            delay = self.config.base_delay_seconds

        # Apply maximum delay cap
        delay = min(delay, self.config.max_delay_seconds)

        # Add jitter to prevent thundering herd
        if self.config.jitter > 0:
            jitter_amount = delay * self.config.jitter
            jitter = random.uniform(-jitter_amount, jitter_amount)
            delay += jitter

        # Ensure non-negative
        delay = max(0, delay)

        return delay

    def _get_fibonacci(self, n: int) -> int:
        """
        Get nth Fibonacci number (cached).

        Args:
            n: Position in Fibonacci sequence

        Returns:
            Fibonacci number at position n
        """
        while len(self._fib_cache) <= n:
            self._fib_cache.append(
                self._fib_cache[-1] + self._fib_cache[-2]
            )

        return self._fib_cache[n]

    def get_metrics(self) -> dict:
        """
        Get retry metrics.

        Returns:
            Dictionary with metrics
        """
        success_rate = 0.0
        if self.metrics.total_attempts > 0:
            success_rate = (
                (self.metrics.total_attempts - self.metrics.failed_retries) /
                self.metrics.total_attempts * 100
            )

        return {
            "total_attempts": self.metrics.total_attempts,
            "successful_retries": self.metrics.successful_retries,
            "failed_retries": self.metrics.failed_retries,
            "success_rate_percent": round(success_rate, 2),
            "total_delay_seconds": round(self.metrics.total_delay_seconds, 2),
            "max_attempts_reached": self.metrics.max_attempts_reached,
            "avg_delay_per_retry": (
                round(self.metrics.total_delay_seconds / self.metrics.successful_retries, 2)
                if self.metrics.successful_retries > 0 else 0.0
            )
        }

    def reset_metrics(self):
        """Reset retry metrics."""
        self.metrics = RetryMetrics()


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
):
    """
    Convenience decorator for adding retry logic.

    Usage:
        @with_retry(max_retries=5, base_delay=2.0)
        async def my_function():
            pass
    """
    config = RetryConfig(
        max_retries=max_retries,
        base_delay_seconds=base_delay,
        strategy=strategy
    )
    handler = RetryHandler(config)
    return handler.retry
