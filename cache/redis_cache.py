"""
Redis Cache Backend - Professional distributed caching.

This provides a production-ready Redis cache backend for the LLM orchestrator.

Benefits over MemoryCache:
- Persistent (survives restarts)
- Shared across multiple processes/servers
- Scalable (can handle millions of entries)
- Built-in TTL support
- Atomic operations
- Can be used in distributed systems

Usage:
    from llm_orchestrator.cache import RedisCache, LLMCache

    # Create Redis backend
    redis_backend = RedisCache(
        host="localhost",
        port=6379,
        db=0,
        password="your-password",  # Optional
        prefix="llm_cache:"  # Optional key prefix
    )

    # Use with LLM cache
    cache = LLMCache(backend=redis_backend, ttl_seconds=3600)
"""

import json
import logging
from typing import Optional, Dict, Any

from .cache import CacheBackend

logger = logging.getLogger(__name__)


class RedisCache(CacheBackend):
    """
    Redis-based cache backend.

    Features:
    - Persistent storage
    - Shared across processes
    - Automatic TTL (expiration)
    - Connection pooling
    - Error resilience
    - Optional key prefix (namespace)

    Example:
        # Simple usage
        cache = RedisCache()

        # With authentication
        cache = RedisCache(
            host="redis.example.com",
            port=6379,
            password="secret",
            db=0
        )

        # With connection pool
        cache = RedisCache(
            host="localhost",
            max_connections=50,
            socket_timeout=5
        )
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        username: Optional[str] = None,
        prefix: str = "llm_cache:",
        max_connections: int = 50,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
        decode_responses: bool = True,
    ):
        """
        Initialize Redis cache.

        Args:
            host: Redis server hostname
            port: Redis server port
            db: Redis database number (0-15)
            password: Redis password (optional)
            username: Redis username (Redis 6+, optional)
            prefix: Key prefix for namespacing (e.g., "llm_cache:")
            max_connections: Max connections in pool
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Connection timeout in seconds
            decode_responses: Decode responses to strings (recommended)
        """
        try:
            import redis
        except ImportError:
            raise ImportError(
                "redis package is required for RedisCache. "
                "Install with: pip install redis"
            )

        self._prefix = prefix
        self._redis_available = True

        try:
            # Create connection pool for better performance
            pool = redis.ConnectionPool(
                host=host,
                port=port,
                db=db,
                password=password,
                username=username,
                max_connections=max_connections,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                decode_responses=decode_responses,
            )

            # Create Redis client
            self._client = redis.Redis(connection_pool=pool)

            # Test connection
            self._client.ping()
            logger.info(f"Redis cache connected to {host}:{port} (db={db})")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            logger.warning("Falling back to in-memory cache")
            self._redis_available = False
            # Fallback to dict (graceful degradation)
            self._fallback_cache: Dict[str, str] = {}

    def _prefixed_key(self, key: str) -> str:
        """Add prefix to key for namespacing"""
        return f"{self._prefix}{key}"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached value by key.

        Args:
            key: Cache key

        Returns:
            Cached value (dict) or None if not found/expired
        """
        if not self._redis_available:
            # Fallback mode
            json_str = self._fallback_cache.get(key)
            return json.loads(json_str) if json_str else None

        try:
            prefixed_key = self._prefixed_key(key)
            value = self._client.get(prefixed_key)

            if value is None:
                return None

            # Deserialize JSON
            return json.loads(value)

        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    def set(self, key: str, value: Dict[str, Any], ttl_seconds: int):
        """
        Store value with expiration time.

        Args:
            key: Cache key
            value: Value to cache (will be JSON-serialized)
            ttl_seconds: Time to live in seconds
        """
        if not self._redis_available:
            # Fallback mode (no TTL in fallback)
            self._fallback_cache[key] = json.dumps(value)
            return

        try:
            prefixed_key = self._prefixed_key(key)

            # Serialize to JSON
            json_str = json.dumps(value)

            # Store with TTL (Redis handles expiration automatically)
            self._client.setex(prefixed_key, ttl_seconds, json_str)

        except Exception as e:
            logger.error(f"Redis set error: {e}")

    def delete(self, key: str):
        """
        Remove cached value.

        Args:
            key: Cache key to delete
        """
        if not self._redis_available:
            self._fallback_cache.pop(key, None)
            return

        try:
            prefixed_key = self._prefixed_key(key)
            self._client.delete(prefixed_key)

        except Exception as e:
            logger.error(f"Redis delete error: {e}")

    def clear(self):
        """
        Clear all cached items with our prefix.

        Note: Only deletes keys matching our prefix to avoid
        deleting other application's data.
        """
        if not self._redis_available:
            self._fallback_cache.clear()
            return

        try:
            # Find all keys with our prefix
            pattern = f"{self._prefix}*"
            keys = list(self._client.scan_iter(match=pattern, count=100))

            if keys:
                # Delete in batches for efficiency
                self._client.delete(*keys)
                logger.info(f"Cleared {len(keys)} cache entries")

        except Exception as e:
            logger.error(f"Redis clear error: {e}")

    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key to check

        Returns:
            True if key exists and not expired
        """
        if not self._redis_available:
            return key in self._fallback_cache

        try:
            prefixed_key = self._prefixed_key(key)
            return bool(self._client.exists(prefixed_key))

        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False

    # ══════════════════════════════════════════════════════════════════════
    # Additional Redis-specific methods
    # ══════════════════════════════════════════════════════════════════════

    def get_ttl(self, key: str) -> Optional[int]:
        """
        Get remaining time to live for a key.

        Args:
            key: Cache key

        Returns:
            Seconds until expiration, or None if key doesn't exist
        """
        if not self._redis_available:
            return None

        try:
            prefixed_key = self._prefixed_key(key)
            ttl = self._client.ttl(prefixed_key)

            # Redis returns -2 if key doesn't exist, -1 if no expiration
            if ttl < 0:
                return None

            return ttl

        except Exception as e:
            logger.error(f"Redis get_ttl error: {e}")
            return None

    def extend_ttl(self, key: str, additional_seconds: int):
        """
        Extend TTL of an existing key.

        Args:
            key: Cache key
            additional_seconds: Seconds to add to current TTL
        """
        if not self._redis_available:
            return

        try:
            prefixed_key = self._prefixed_key(key)
            current_ttl = self._client.ttl(prefixed_key)

            if current_ttl > 0:
                new_ttl = current_ttl + additional_seconds
                self._client.expire(prefixed_key, new_ttl)

        except Exception as e:
            logger.error(f"Redis extend_ttl error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get Redis cache statistics.

        Returns:
            Dict with memory usage, key count, etc.
        """
        if not self._redis_available:
            return {
                "available": False,
                "fallback_size": len(self._fallback_cache),
            }

        try:
            # Get Redis info
            info = self._client.info("memory")
            stats = self._client.info("stats")

            # Count keys with our prefix
            pattern = f"{self._prefix}*"
            key_count = sum(1 for _ in self._client.scan_iter(match=pattern, count=100))

            return {
                "available": True,
                "key_count": key_count,
                "used_memory_mb": info.get("used_memory", 0) / (1024 * 1024),
                "used_memory_peak_mb": info.get("used_memory_peak", 0) / (1024 * 1024),
                "total_commands_processed": stats.get("total_commands_processed", 0),
                "hit_rate_percent": round(
                    stats.get("keyspace_hits", 0)
                    / max(stats.get("keyspace_hits", 0) + stats.get("keyspace_misses", 0), 1)
                    * 100,
                    2,
                ),
            }

        except Exception as e:
            logger.error(f"Redis get_stats error: {e}")
            return {"available": False, "error": str(e)}

    def ping(self) -> bool:
        """
        Test Redis connection.

        Returns:
            True if Redis is available and responding
        """
        if not self._redis_available:
            return False

        try:
            return self._client.ping()
        except Exception as e:
            logger.error(f"Redis ping error: {e}")
            return False

    def close(self):
        """Close Redis connection pool"""
        if self._redis_available and hasattr(self, "_client"):
            try:
                self._client.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")

    def __del__(self):
        """Cleanup on deletion"""
        self.close()
