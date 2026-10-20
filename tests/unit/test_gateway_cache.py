from decimal import Decimal
from typing import Any

import pytest

from agentic_ops.gateway.cache import InMemoryCache, cache_key, get_or_compute


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class Computer:
    """Counts how often the expensive computation actually runs."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self) -> dict[str, Any]:
        self.calls += 1
        return {'answer': 42}


def test_key_is_stable_across_dict_ordering() -> None:
    first = cache_key(namespace='default', request={'a': 1, 'b': {'x': 1, 'y': 2}})
    second = cache_key(namespace='default', request={'b': {'y': 2, 'x': 1}, 'a': 1})
    assert first == second


def test_key_differs_by_content_and_namespace() -> None:
    base = cache_key(namespace='default', request={'prompt': 'hello'})
    assert cache_key(namespace='default', request={'prompt': 'hullo'}) != base
    assert cache_key(namespace='fast', request={'prompt': 'hello'}) != base


def test_key_is_a_sha256_hex_digest() -> None:
    key = cache_key(namespace='default', request={})
    assert len(key) == 64
    assert set(key) <= set('0123456789abcdef')


def test_key_rejects_non_json_safe_requests() -> None:
    with pytest.raises(TypeError):
        cache_key(namespace='default', request={'qty': Decimal('1.5')})


async def test_miss_returns_none() -> None:
    assert await InMemoryCache().get('absent') is None


async def test_stored_values_are_isolated_from_caller_mutation() -> None:
    cache = InMemoryCache()
    original: dict[str, Any] = {'text': 'hi', 'nested': {'n': 1}}
    await cache.set('k', original)
    original['text'] = 'changed'
    fetched = await cache.get('k')
    assert fetched == {'text': 'hi', 'nested': {'n': 1}}
    assert fetched is not None
    fetched['nested']['n'] = 99
    assert await cache.get('k') == {'text': 'hi', 'nested': {'n': 1}}


async def test_entries_expire_after_ttl() -> None:
    clock = FakeClock()
    cache = InMemoryCache(clock=clock)
    await cache.set('k', {'v': 1}, ttl=10.0)
    clock.now = 9.9
    assert await cache.get('k') == {'v': 1}
    clock.now = 10.0
    assert await cache.get('k') is None


async def test_entries_without_ttl_never_expire() -> None:
    clock = FakeClock()
    cache = InMemoryCache(clock=clock)
    await cache.set('k', {'v': 1})
    clock.now = 1e9
    assert await cache.get('k') == {'v': 1}


async def test_least_recently_used_entry_is_evicted() -> None:
    cache = InMemoryCache(max_entries=2)
    await cache.set('a', {'v': 1})
    await cache.set('b', {'v': 2})
    await cache.get('a')
    await cache.set('c', {'v': 3})
    assert await cache.get('b') is None
    assert await cache.get('a') == {'v': 1}
    assert await cache.get('c') == {'v': 3}


async def test_get_or_compute_computes_once_then_hits() -> None:
    cache = InMemoryCache()
    computer = Computer()
    first = await get_or_compute(cache, 'k', computer)
    second = await get_or_compute(cache, 'k', computer)
    assert (first.hit, second.hit) == (False, True)
    assert first.value == second.value == {'answer': 42}
    assert computer.calls == 1


async def test_get_or_compute_honours_ttl() -> None:
    clock = FakeClock()
    cache = InMemoryCache(clock=clock)
    computer = Computer()
    await get_or_compute(cache, 'k', computer, ttl=5.0)
    clock.now = 6.0
    result = await get_or_compute(cache, 'k', computer, ttl=5.0)
    assert not result.hit
    assert computer.calls == 2


async def test_failed_compute_is_not_cached() -> None:
    cache = InMemoryCache()

    async def boom() -> dict[str, Any]:
        raise RuntimeError('provider down')

    with pytest.raises(RuntimeError):
        await get_or_compute(cache, 'k', boom)
    assert await cache.get('k') is None
