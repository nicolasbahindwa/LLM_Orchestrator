import time

from llm_orchestrator.cache import LLMCache, MemoryCache
from llm_orchestrator.models import LLMRequest, LLMResponse


def test_memory_cache_ttl_expires() -> None:
    cache = MemoryCache()
    cache.set("k", {"v": 1}, ttl_seconds=1)
    assert cache.get("k") == {"v": 1}

    time.sleep(1.05)
    assert cache.get("k") is None


def test_memory_cache_evicts_oldest() -> None:
    cache = MemoryCache(max_size=2)
    cache.set("k1", {"v": 1}, ttl_seconds=60)
    cache.set("k2", {"v": 2}, ttl_seconds=60)
    cache.set("k3", {"v": 3}, ttl_seconds=60)

    assert cache.get("k1") is None
    assert cache.get("k2") == {"v": 2}
    assert cache.get("k3") == {"v": 3}


def test_llm_cache_roundtrip_marks_cached() -> None:
    llm_cache = LLMCache(backend=MemoryCache(), ttl_seconds=60)
    req = LLMRequest(prompt="hello")
    resp = LLMResponse(
        content="world",
        provider="fake",
        model="m",
        request_id=req.request_id,
        success=True,
    )

    llm_cache.set(req, resp)
    cached = llm_cache.get(req)

    assert cached is not None
    assert cached.cached is True
    assert cached.content == "world"


def test_llm_cache_does_not_store_failed_response() -> None:
    llm_cache = LLMCache(backend=MemoryCache(), ttl_seconds=60)
    req = LLMRequest(prompt="hello")
    failed = LLMResponse(
        content="",
        provider="fake",
        model="m",
        request_id=req.request_id,
        success=False,
        error_message="boom",
    )

    llm_cache.set(req, failed)
    assert llm_cache.get(req) is None