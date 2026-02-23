# Redis Cache Usage Guide

## Overview

The LLM Orchestrator now supports professional Redis caching for production deployments. Redis provides:

- **Persistent caching** - Survives restarts
- **Distributed caching** - Shared across multiple processes/servers
- **Scalability** - Handles millions of entries efficiently
- **Built-in TTL** - Automatic expiration
- **Atomic operations** - Thread-safe

## Installation

```bash
# Install Redis client
pip install redis

# Or install with orchestrator extras (future)
pip install llm-orchestrator[redis]
```

## Quick Start

### 1. Using orchestrator.yaml

```yaml
cache:
  enabled: true
  backend: redis  # Switch from memory to redis
  ttl: 3600

  # Redis connection settings
  redis_host: localhost
  redis_port: 6379
  redis_db: 0
  redis_password: ${REDIS_PASSWORD}  # Optional
  redis_prefix: "llm_cache:"
  redis_max_connections: 50
  redis_socket_timeout: 5
```

```python
from llm_orchestrator import create_orchestrator

# Redis cache automatically configured from YAML
orchestrator = create_orchestrator("orchestrator.yaml")
```

### 2. Programmatic Configuration

```python
from llm_orchestrator import create_orchestrator
from llm_orchestrator.cache import RedisCache, LLMCache

# Create Redis backend manually
redis_backend = RedisCache(
    host="localhost",
    port=6379,
    db=0,
    password="your-password",  # Optional
    prefix="llm_cache:"
)

# Create cache with Redis backend
cache = LLMCache(
    backend=redis_backend,
    ttl_seconds=3600
)

# Use with orchestrator
orchestrator = create_orchestrator("orchestrator.yaml")
orchestrator.cache = cache
```

## Configuration Options

### Basic Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `redis_host` | `localhost` | Redis server hostname |
| `redis_port` | `6379` | Redis server port |
| `redis_db` | `0` | Redis database number (0-15) |
| `redis_password` | `None` | Password for authentication |
| `redis_username` | `None` | Username (Redis 6+ only) |

### Advanced Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `redis_prefix` | `llm_cache:` | Key prefix for namespacing |
| `redis_max_connections` | `50` | Max connections in pool |
| `redis_socket_timeout` | `5` | Socket timeout (seconds) |

## Features

### 1. Automatic Fallback

If Redis is unavailable, the cache automatically falls back to in-memory storage:

```python
redis_cache = RedisCache(host="unreachable-host")
# Logs warning and uses in-memory fallback
# Your app continues working!
```

### 2. Key Namespacing

Use prefixes to avoid conflicts with other applications:

```python
# Production cache
prod_cache = RedisCache(prefix="llm_prod:")

# Development cache
dev_cache = RedisCache(prefix="llm_dev:")

# Per-tenant caching
tenant_cache = RedisCache(prefix=f"tenant_{tenant_id}:")
```

### 3. Connection Pooling

Efficient connection reuse for high-performance:

```python
cache = RedisCache(
    max_connections=100,  # High-traffic production
    socket_timeout=10,
    socket_connect_timeout=5
)
```

### 4. Redis-Specific Methods

```python
from llm_orchestrator.cache import RedisCache

cache = RedisCache()

# Get remaining TTL
ttl = cache.get_ttl("some_key")
print(f"Expires in {ttl} seconds")

# Extend expiration time
cache.extend_ttl("some_key", additional_seconds=3600)

# Get cache statistics
stats = cache.get_stats()
print(f"Cache hit rate: {stats['hit_rate_percent']}%")
print(f"Memory used: {stats['used_memory_mb']} MB")
print(f"Total keys: {stats['key_count']}")

# Test connection
if cache.ping():
    print("Redis is healthy!")
```

## Production Deployment

### Docker Compose Example

```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}

  orchestrator:
    build: .
    environment:
      REDIS_PASSWORD: ${REDIS_PASSWORD}
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    depends_on:
      - redis

volumes:
  redis_data:
```

### Environment Variables

```bash
# .env file
REDIS_PASSWORD=your-secure-password
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### Remote Redis (Production)

```yaml
cache:
  backend: redis
  redis_host: redis.production.example.com
  redis_port: 6379
  redis_db: 0
  redis_password: ${REDIS_PASSWORD}
  redis_prefix: "prod_llm:"
  redis_max_connections: 100
```

## Performance Tuning

### High-Traffic Scenarios

```python
cache = RedisCache(
    max_connections=200,      # More connections
    socket_timeout=10,        # Longer timeout
    redis_prefix="ht_llm:",   # Separate namespace
)
```

### Multi-Tenant Setup

```python
def get_tenant_cache(tenant_id: str):
    return RedisCache(
        prefix=f"tenant_{tenant_id}:",
        db=hash(tenant_id) % 16,  # Distribute across DBs
    )
```

### Monitoring Cache Performance

```python
import time

# Get initial stats
stats_before = cache.get_stats()

# ... run your workload ...

# Get final stats
stats_after = cache.get_stats()

# Calculate metrics
hit_rate = stats_after['hit_rate_percent']
memory_mb = stats_after['used_memory_mb']
key_count = stats_after['key_count']

print(f"Cache Hit Rate: {hit_rate}%")
print(f"Memory Usage: {memory_mb} MB")
print(f"Cached Keys: {key_count}")
```

## Troubleshooting

### Connection Refused

```
Error: Redis connection failed - Connection refused
```

**Solution**: Make sure Redis is running:

```bash
# Start Redis (macOS)
brew services start redis

# Start Redis (Linux)
sudo systemctl start redis

# Start Redis (Docker)
docker run -d -p 6379:6379 redis:7-alpine
```

### Authentication Failed

```
Error: NOAUTH Authentication required
```

**Solution**: Provide password in config:

```yaml
cache:
  redis_password: ${REDIS_PASSWORD}
```

### Memory Issues

```
Error: OOM command not allowed when used memory > 'maxmemory'
```

**Solution**: Configure Redis eviction policy:

```bash
# redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru  # Evict least recently used
```

## Migration from Memory Cache

### Before (Memory Cache)

```yaml
cache:
  enabled: true
  backend: memory
  ttl: 3600
  max_size: 1000
```

### After (Redis Cache)

```yaml
cache:
  enabled: true
  backend: redis  # Just change this!
  ttl: 3600

  # Add Redis settings
  redis_host: localhost
  redis_port: 6379
  redis_db: 0
```

**No code changes required!** The orchestrator automatically uses Redis.

## Best Practices

1. **Use key prefixes** to avoid conflicts with other apps
2. **Set appropriate TTL** based on your use case
3. **Monitor cache hit rates** to optimize performance
4. **Use connection pooling** for high-traffic scenarios
5. **Enable persistence** (AOF) for production Redis
6. **Secure with passwords** in production
7. **Use separate Redis DB** for dev/staging/prod

## Example: Full Production Setup

```python
from llm_orchestrator import create_orchestrator
from llm_orchestrator.models import LLMRequest

# Load config (with Redis cache)
orchestrator = create_orchestrator("orchestrator.yaml")

# Make requests (automatically cached in Redis)
response1 = await orchestrator.generate(LLMRequest(
    prompt="What is AI?"
))

# Same request - served from Redis cache (instant!)
response2 = await orchestrator.generate(LLMRequest(
    prompt="What is AI?"
))

# Check cache stats
if orchestrator.cache and hasattr(orchestrator.cache.backend, 'get_stats'):
    stats = orchestrator.cache.backend.get_stats()
    print(f"Cache hit rate: {stats['hit_rate_percent']}%")
    print(f"Saved ${stats.get('cost_saved_usd', 0):.2f} in API costs")
```

## See Also

- [Cache Architecture](cache.py) - Base cache implementation
- [Memory Cache](cache.py) - In-memory cache backend
- [Configuration Guide](../config/README.md) - Full config reference
