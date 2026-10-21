"""Behavioural contract for ResponseCache stores, plus Postgres-specific checks."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from agentic_ops.db.models import CacheEntry
from agentic_ops.db.session import get_sessionmaker
from agentic_ops.gateway.cache import InMemoryCache, ResponseCache, get_or_compute
from agentic_ops.gateway.cache_pg import PostgresCache

_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)
integration = pytest.mark.integration


class FakeClock:
    """Deterministic time source shared by every implementation under test."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def utc_clock(clock: FakeClock) -> Callable[[], datetime]:
    """Adapt the float clock to the timezone-aware datetimes the Postgres store expects."""

    def now() -> datetime:
        return _EPOCH + timedelta(seconds=clock())

    return now


@dataclass(frozen=True)
class CacheCase:
    cache: ResponseCache
    clock: FakeClock


def _key() -> str:
    return uuid4().hex


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def session_factory(async_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return get_sessionmaker(async_engine)


@pytest.fixture(params=['memory', pytest.param('postgres', marks=pytest.mark.integration)])
def cache_case(request: pytest.FixtureRequest, clock: FakeClock) -> CacheCase:
    """Every contract test runs against both stores; the DB is requested only for Postgres."""
    if request.param == 'memory':
        return CacheCase(cache=InMemoryCache(clock=clock), clock=clock)
    engine: AsyncEngine = request.getfixturevalue('async_engine')
    store = PostgresCache(get_sessionmaker(engine), clock=utc_clock(clock))
    return CacheCase(cache=store, clock=clock)


# --- Shared contract: InMemoryCache and PostgresCache must behave identically ---


async def test_round_trip(cache_case: CacheCase) -> None:
    key = _key()
    await cache_case.cache.set(key, {'answer': 42})
    assert await cache_case.cache.get(key) == {'answer': 42}


async def test_miss_returns_none(cache_case: CacheCase) -> None:
    assert await cache_case.cache.get(_key()) is None


async def test_set_overwrites_existing_value(cache_case: CacheCase) -> None:
    key = _key()
    await cache_case.cache.set(key, {'v': 1})
    await cache_case.cache.set(key, {'v': 2})
    assert await cache_case.cache.get(key) == {'v': 2}


async def test_entry_expires_when_ttl_elapses(cache_case: CacheCase) -> None:
    key = _key()
    await cache_case.cache.set(key, {'v': 1}, ttl=60)
    cache_case.clock.advance(59)
    assert await cache_case.cache.get(key) == {'v': 1}
    cache_case.clock.advance(1)
    assert await cache_case.cache.get(key) is None


async def test_entry_without_ttl_never_expires(cache_case: CacheCase) -> None:
    key = _key()
    await cache_case.cache.set(key, {'v': 1})
    cache_case.clock.advance(10_000_000)
    assert await cache_case.cache.get(key) == {'v': 1}


async def test_overwrite_resets_ttl(cache_case: CacheCase) -> None:
    key = _key()
    await cache_case.cache.set(key, {'v': 1}, ttl=10)
    cache_case.clock.advance(8)
    await cache_case.cache.set(key, {'v': 2}, ttl=10)
    cache_case.clock.advance(8)
    assert await cache_case.cache.get(key) == {'v': 2}


async def test_returned_value_is_isolated_from_the_store(cache_case: CacheCase) -> None:
    key = _key()
    await cache_case.cache.set(key, {'items': [1, 2]})
    first = await cache_case.cache.get(key)
    assert first is not None
    first['items'].append(3)
    assert await cache_case.cache.get(key) == {'items': [1, 2]}


async def test_stored_value_is_isolated_from_the_caller(cache_case: CacheCase) -> None:
    key = _key()
    value: dict[str, Any] = {'items': [1, 2]}
    await cache_case.cache.set(key, value)
    value['items'].append(3)
    assert await cache_case.cache.get(key) == {'items': [1, 2]}


async def test_nested_json_values_round_trip(cache_case: CacheCase) -> None:
    key = _key()
    value: dict[str, Any] = {'a': [1, 2.5, None, True], 'b': {'c': 'é'}}
    await cache_case.cache.set(key, value)
    assert await cache_case.cache.get(key) == value


async def test_empty_value_is_a_hit_not_a_miss(cache_case: CacheCase) -> None:
    key = _key()
    await cache_case.cache.set(key, {})
    assert await cache_case.cache.get(key) == {}


async def test_get_or_compute_computes_once(cache_case: CacheCase) -> None:
    calls: list[int] = []

    async def compute() -> dict[str, Any]:
        calls.append(1)
        return {'text': 'hello'}

    key = _key()
    first = await get_or_compute(cache_case.cache, key, compute)
    second = await get_or_compute(cache_case.cache, key, compute)
    assert (first.hit, second.hit) == (False, True)
    assert second.value == {'text': 'hello'}
    assert len(calls) == 1


async def test_get_or_compute_recomputes_after_expiry(cache_case: CacheCase) -> None:
    calls: list[int] = []

    async def compute() -> dict[str, Any]:
        calls.append(1)
        return {'n': len(calls)}

    key = _key()
    await get_or_compute(cache_case.cache, key, compute, ttl=30)
    cache_case.clock.advance(30)
    result = await get_or_compute(cache_case.cache, key, compute, ttl=30)
    assert result.hit is False
    assert result.value == {'n': 2}


async def test_failed_compute_is_not_cached(cache_case: CacheCase) -> None:
    async def boom() -> dict[str, Any]:
        raise RuntimeError('provider down')

    async def ok() -> dict[str, Any]:
        return {'ok': True}

    key = _key()
    with pytest.raises(RuntimeError):
        await get_or_compute(cache_case.cache, key, boom)
    result = await get_or_compute(cache_case.cache, key, ok)
    assert result.hit is False
    assert result.value == {'ok': True}


# --- Postgres-specific behaviour ---


@integration
async def test_entries_survive_a_new_store_instance(
    session_factory: async_sessionmaker[AsyncSession], clock: FakeClock
) -> None:
    """A fresh instance stands in for a restarted process: state lives in the database."""
    key = _key()
    await PostgresCache(session_factory, clock=utc_clock(clock)).set(key, {'v': 1})
    restarted = PostgresCache(session_factory, clock=utc_clock(clock))
    assert await restarted.get(key) == {'v': 1}


@integration
async def test_upsert_keeps_one_committed_row(
    session_factory: async_sessionmaker[AsyncSession], clock: FakeClock
) -> None:
    cache = PostgresCache(session_factory, clock=utc_clock(clock))
    key = _key()
    await cache.set(key, {'v': 1})
    await cache.set(key, {'v': 2})
    async with session_factory() as session:
        stmt = select(func.count()).select_from(CacheEntry).where(CacheEntry.key == key)
        assert await session.scalar(stmt) == 1
        row = await session.get(CacheEntry, key)
        assert row is not None
        assert row.value == {'v': 2}


@integration
async def test_purge_expired_deletes_only_expired_rows(
    session_factory: async_sessionmaker[AsyncSession], clock: FakeClock
) -> None:
    cache = PostgresCache(session_factory, clock=utc_clock(clock))
    short, forever, lasting = _key(), _key(), _key()
    await cache.set(short, {'v': 1}, ttl=10)
    await cache.set(forever, {'v': 2})
    await cache.set(lasting, {'v': 3}, ttl=100)
    clock.advance(50)
    deleted = await cache.purge_expired()
    assert deleted >= 1
    async with session_factory() as session:
        assert await session.get(CacheEntry, short) is None
        assert await session.get(CacheEntry, forever) is not None
        assert await session.get(CacheEntry, lasting) is not None
