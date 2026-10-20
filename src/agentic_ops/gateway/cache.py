import copy
import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

Clock = Callable[[], float]


def cache_key(*, namespace: str, request: Mapping[str, Any]) -> str:
    """Stable SHA-256 key for a request within a namespace.

    The request is serialised as canonical JSON (sorted keys, fixed separators), so
    dict ordering never changes the key. Values that JSON cannot represent, such as
    Decimal or datetime, raise TypeError (NaN raises ValueError): the caller must pass
    JSON-safe data, for example model_dump(mode='json'), so no meaning is coerced away.
    """
    canonical = json.dumps(
        {'namespace': namespace, 'request': dict(request)},
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


class ResponseCache(Protocol):
    """Storage interface; the Postgres-backed store will implement the same two methods."""

    async def get(self, key: str) -> dict[str, Any] | None: ...

    async def set(self, key: str, value: dict[str, Any], *, ttl: float | None = None) -> None: ...


class InMemoryCache:
    """Bounded LRU cache with optional per-entry TTL, for one process."""

    def __init__(self, *, max_entries: int = 256, clock: Clock = time.monotonic) -> None:
        if max_entries < 1:
            raise ValueError('max_entries must be at least 1')
        self._max_entries = max_entries
        self._clock = clock
        self._entries: OrderedDict[str, tuple[float | None, dict[str, Any]]] = OrderedDict()

    async def get(self, key: str) -> dict[str, Any] | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at is not None and self._clock() >= expires_at:
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return copy.deepcopy(value)

    async def set(self, key: str, value: dict[str, Any], *, ttl: float | None = None) -> None:
        expires_at = None if ttl is None else self._clock() + ttl
        self._entries[key] = (expires_at, copy.deepcopy(value))
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)


@dataclass(frozen=True)
class CacheResult:
    """A value plus whether it came from the cache; hit feeds the cache-hit-rate metric."""

    value: dict[str, Any]
    hit: bool


async def get_or_compute(
    cache: ResponseCache,
    key: str,
    compute: Callable[[], Awaitable[dict[str, Any]]],
    *,
    ttl: float | None = None,
) -> CacheResult:
    """Return the cached value, or compute, store and return it.

    Only successful computations are stored: an exception propagates before the write,
    so an outage is never replayed from the cache for the length of the TTL.
    """
    cached = await cache.get(key)
    if cached is not None:
        return CacheResult(value=cached, hit=True)
    value = await compute()
    await cache.set(key, value, ttl=ttl)
    return CacheResult(value=value, hit=False)
