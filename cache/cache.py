"""
Cache System - Avoid duplicate LLM requests.

This file implements caching to prevent sending the same request to expensive
LLM APIs multiple times.

Why cache exists:
- Save money: Don't pay for the same response twice
- Save time: Cached responses return instantly
- Reduce load: Less strain on LLM providers

How it works:
1. Before calling LLM API, check if we've seen this request before
2. If yes → return cached response immediately
3. If no → call API, then save response to cache for next time
"""

import hashlib
import json
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

from ..models import LLMRequest, LLMResponse


class CacheBackend(ABC):
    """
    Abstract base for cache storage.

    Allows different storage backends:
    - Memory (fast, lost on restart)
    - Redis (persistent, shared across instances)
    - File (simple, persistent)
    """

    @abstractmethod
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached value by key"""
        pass

    @abstractmethod
    def set(self, key: str, value: Dict[str, Any], ttl_seconds: int):
        """Store value with expiration time"""
        pass

    @abstractmethod
    def delete(self, key: str):
        """Remove cached value"""
        pass

    @abstractmethod
    def clear(self):
        """Clear entire cache"""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass


class MemoryCache(CacheBackend):
    """
    In-memory cache using Python dictionary.

    Pros:
    - Very fast (no I/O)
    - Simple, no dependencies

    Cons:
    - Lost on restart
    - Not shared across processes
    - Limited by RAM
    """

    def __init__(self, max_size: int = 10000):
        """
        Initialize memory cache.

        Args:
            max_size: Maximum number of cached items (prevents memory overflow)
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_size = max_size

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached value, checking expiration"""
        if key not in self._cache:
            return None

        entry = self._cache[key]

        # Check if expired
        if entry["expires_at"] < time.time():
            del self._cache[key]
            return None

        return entry["value"]

    def set(self, key: str, value: Dict[str, Any], ttl_seconds: int):
        """Store value with TTL (Time To Live)"""
        # Evict oldest if at max size
        if len(self._cache) >= self._max_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]["created_at"])
            del self._cache[oldest_key]

        self._cache[key] = {
            "value": value,
            "created_at": time.time(),
            "expires_at": time.time() + ttl_seconds,
        }

    def delete(self, key: str):
        """Remove specific key"""
        self._cache.pop(key, None)

    def clear(self):
        """Clear all cached items"""
        self._cache.clear()

    def exists(self, key: str) -> bool:
        """Check if key exists and not expired"""
        return self.get(key) is not None

    def size(self) -> int:
        """Get current cache size"""
        return len(self._cache)


class LLMCache:
    """
    High-level cache for LLM requests and responses.

    Handles:
    - Request hashing (convert request to unique key)
    - Response serialization (convert to/from dict for storage)
    - Cache hit/miss tracking (metrics)
    - TTL management (expiration)

    Example:
        cache = LLMCache(backend=MemoryCache(), ttl_seconds=3600)

        # Try to get from cache
        cached = cache.get(request)
        if cached:
            return cached  # Cache hit!

        # Cache miss - call API
        response = await provider.generate(request)

        # Save for next time
        cache.set(request, response)
    """

    def __init__(self, backend: CacheBackend, ttl_seconds: int = 3600):
        """
        Initialize LLM cache.

        Args:
            backend: Storage backend (MemoryCache, RedisCache, etc.)
            ttl_seconds: How long to cache responses (default: 1 hour)
        """
        self._backend = backend
        self._ttl_seconds = ttl_seconds

        # Metrics
        self._hits = 0
        self._misses = 0
        self._total_requests = 0

    def get(self, request: LLMRequest) -> Optional[LLMResponse]:
        """
        Try to get cached response for request.

        Args:
            request: LLM request to look up

        Returns:
            Cached LLMResponse if found, None otherwise
        """
        self._total_requests += 1

        # Generate cache key from request
        cache_key = self._generate_key(request)

        # Try to get from backend
        cached_data = self._backend.get(cache_key)

        if cached_data is None:
            self._misses += 1
            return None

        # Cache hit! Deserialize response
        self._hits += 1
        response = self._deserialize_response(cached_data)

        # Mark as cached
        response.cached = True

        return response

    def set(self, request: LLMRequest, response: LLMResponse):
        """
        Cache a response for future requests.

        Args:
            request: The request that was made
            response: The response to cache
        """
        # Only cache successful responses
        if not response.success:
            return

        cache_key = self._generate_key(request)
        serialized = self._serialize_response(response)

        self._backend.set(cache_key, serialized, self._ttl_seconds)

    def delete(self, request: LLMRequest):
        """
        Remove cached response for a request.

        Args:
            request: Request to remove from cache
        """
        cache_key = self._generate_key(request)
        self._backend.delete(cache_key)

    def clear(self):
        """Clear entire cache"""
        self._backend.clear()
        self._hits = 0
        self._misses = 0
        self._total_requests = 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.

        Returns:
            Dict with hit rate, miss rate, total requests
        """
        hit_rate = (self._hits / self._total_requests * 100) if self._total_requests > 0 else 0
        miss_rate = (self._misses / self._total_requests * 100) if self._total_requests > 0 else 0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "total_requests": self._total_requests,
            "hit_rate_percent": round(hit_rate, 2),
            "miss_rate_percent": round(miss_rate, 2),
        }

    # ══════════════════════════════════════════════════════════════════════
    # PRIVATE HELPER METHODS
    # ══════════════════════════════════════════════════════════════════════

    def _generate_key(self, request: LLMRequest) -> str:
        """
        Generate unique cache key from request.

        The key is based on:
        - Prompt text
        - Max tokens
        - Temperature
        - System prompt

        Two identical requests will have the same key.

        Args:
            request: LLM request

        Returns:
            Unique string key (SHA-256 hash)
        """
        # Create deterministic representation of request
        key_data = {
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "system_prompt": request.system_prompt or "",
        }

        # Convert to JSON (sorted keys for consistency)
        json_str = json.dumps(key_data, sort_keys=True)

        # Hash to fixed-length key
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _serialize_response(self, response: LLMResponse) -> Dict[str, Any]:
        """
        Convert LLMResponse to dictionary for storage.

        Args:
            response: Response to serialize

        Returns:
            Dictionary representation
        """
        return {
            "content": response.content,
            "provider": response.provider,
            "model": response.model,
            "request_id": response.request_id,
            "success": response.success,
            "error_message": response.error_message,
            "usage": response.usage,
            "latency_ms": response.latency_ms,
            "raw_response": response.raw_response,
            "cached": False,  # Will be set to True when retrieved
            "fallback_used": response.fallback_used,
            "attempted_providers": response.attempted_providers,
            "timestamp": response.timestamp.isoformat(),
        }

    def _deserialize_response(self, data: Dict[str, Any]) -> LLMResponse:
        """
        Convert dictionary back to LLMResponse.

        Args:
            data: Serialized response data

        Returns:
            LLMResponse object
        """
        # Parse timestamp
        timestamp = datetime.fromisoformat(data["timestamp"])

        return LLMResponse(
            content=data["content"],
            provider=data["provider"],
            model=data["model"],
            request_id=data["request_id"],
            success=data["success"],
            error_message=data.get("error_message"),
            usage=data.get("usage"),
            latency_ms=data["latency_ms"],
            raw_response=data.get("raw_response"),
            cached=True,  # Mark as cached
            fallback_used=data.get("fallback_used", False),
            attempted_providers=data.get("attempted_providers", []),
            timestamp=timestamp,
        )
