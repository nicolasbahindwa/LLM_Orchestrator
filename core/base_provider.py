"""
Base Provider - Abstract interface for all LLM providers.

This file defines what EVERY provider must implement. It's like a contract
that says "if you want to be a provider in this orchestrator, you must have
these methods."

Why this file exists:
- Consistency: All providers work the same way
- Polymorphism: Router can use any provider without knowing which one
- Easy to add new providers: Just implement this interface
- Type safety: Ensures all providers have required methods
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from ..models import LLMRequest, LLMResponse, ProviderMetrics


class BaseProvider(ABC):
    """
    Abstract base class for all LLM providers.

    Every provider (Anthropic, OpenAI, Google, etc.) must inherit from this
    and implement all the abstract methods.

    Think of this as a template or contract that ensures all providers:
    1. Can generate text (generate method)
    2. Can report their health (get_metrics method)
    3. Have a unique name (name property)
    4. Know which models they support (get_available_models method)

    Example:
        class AnthropicProvider(BaseProvider):
            def __init__(self, api_key: str):
                self.api_key = api_key
                self.name = "anthropic"

            async def generate(self, request: LLMRequest) -> LLMResponse:
                # Call Anthropic API
                ...

            def get_metrics(self) -> ProviderMetrics:
                # Return health metrics
                ...
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize the provider.

        Args:
            name: Unique provider name (e.g., "anthropic", "openai")
            config: Provider-specific configuration (API key, base URL, etc.)
        """
        self._name = name
        self._config = config
        self._metrics = ProviderMetrics(provider_name=name)

    # ══════════════════════════════════════════════════════════════════════
    # PROPERTIES - Basic provider information
    # ══════════════════════════════════════════════════════════════════════

    @property
    def name(self) -> str:
        """Get provider name (e.g., 'anthropic', 'openai')"""
        return self._name

    @property
    def config(self) -> Dict[str, Any]:
        """Get provider configuration"""
        return self._config

    # ══════════════════════════════════════════════════════════════════════
    # ABSTRACT METHODS - Must be implemented by each provider
    # ══════════════════════════════════════════════════════════════════════

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate text using this provider's LLM.

        This is the CORE method - it sends the request to the LLM API
        and returns the response.

        Args:
            request: The LLM request with prompt, parameters, etc.

        Returns:
            LLMResponse with generated text and metadata

        Raises:
            Exception: If the API call fails

        Example Implementation:
            async def generate(self, request: LLMRequest) -> LLMResponse:
                start_time = time.time()

                try:
                    # Call the actual LLM API
                    response = await self.api_client.create_completion(
                        prompt=request.prompt,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature
                    )

                    # Track success
                    latency = (time.time() - start_time) * 1000
                    self._metrics.update_success(latency)

                    return LLMResponse(
                        content=response.text,
                        provider=self.name,
                        model=response.model,
                        request_id=request.request_id,
                        success=True,
                        latency_ms=latency
                    )

                except Exception as e:
                    # Track failure
                    self._metrics.update_error()
                    return LLMResponse(
                        content="",
                        provider=self.name,
                        model="",
                        request_id=request.request_id,
                        success=False,
                        error_message=str(e)
                    )
        """
        pass

    @abstractmethod
    def get_available_models(self) -> list[str]:
        """
        Get list of models this provider supports.

        Returns:
            List of model names (e.g., ["claude-sonnet-4", "claude-opus-4"])

        Example:
            def get_available_models(self) -> list[str]:
                return [
                    "claude-sonnet-4-20250514",
                    "claude-opus-4-20250514",
                    "claude-haiku-4-20250514"
                ]
        """
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate that the provider configuration is correct.

        Checks:
        - API key is present and valid format
        - Required config fields are set
        - Can connect to API (optional health check)

        Returns:
            True if configuration is valid, False otherwise

        Example:
            def validate_config(self) -> bool:
                if not self._config.get("api_key"):
                    return False
                if len(self._config["api_key"]) < 10:
                    return False
                return True
        """
        pass

    # ══════════════════════════════════════════════════════════════════════
    # CONCRETE METHODS - Shared by all providers (already implemented)
    # ══════════════════════════════════════════════════════════════════════

    def get_metrics(self) -> ProviderMetrics:
        """
        Get current health metrics for this provider.

        Returns metrics like:
        - Success rate
        - Average latency
        - Error count
        - Health status

        Returns:
            ProviderMetrics object

        Note: This is NOT abstract - the base class tracks metrics automatically
              when you call update_success() or update_error() in your generate() method.
        """
        return self._metrics

    def is_healthy(self) -> bool:
        """
        Check if provider is healthy and ready to use.

        A provider is healthy if:
        - Configuration is valid
        - Not in circuit breaker open state
        - Success rate is acceptable

        Returns:
            True if healthy, False otherwise
        """
        return (
            self.validate_config() and
            self._metrics.is_healthy()
        )

    def supports_model(self, model: str) -> bool:
        """
        Check if this provider supports a specific model.

        Args:
            model: Model name to check

        Returns:
            True if supported, False otherwise

        Example:
            if provider.supports_model("claude-sonnet-4"):
                # Use this provider
                response = await provider.generate(request)
        """
        return model in self.get_available_models()

    def reset_metrics(self):
        """
        Reset provider metrics to initial state.

        Useful for:
        - Testing
        - After recovering from errors
        - Scheduled maintenance
        """
        self._metrics = ProviderMetrics(provider_name=self._name)

    # ══════════════════════════════════════════════════════════════════════
    # HELPER METHODS - Utilities for subclasses
    # ══════════════════════════════════════════════════════════════════════

    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Safely get a configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self._config.get(key, default)

    def __repr__(self) -> str:
        """String representation for debugging"""
        return f"{self.__class__.__name__}(name='{self.name}', healthy={self.is_healthy()})"
