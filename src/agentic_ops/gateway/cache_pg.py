from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import delete, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agentic_ops.db.models import CacheEntry

Now = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class PostgresCache:
    """ResponseCache backed by Postgres, so entries survive restarts and cold starts.

    Each operation opens its own short session and commits on its own. This is the
    deliberate opposite of record_event: a cache write is an optimisation and must not
    roll back with, or poison, an agent transaction. The clock must return timezone-aware
    datetimes; one clock authority writes and reads every deadline.
    """

    def __init__(
        self, sessions: async_sessionmaker[AsyncSession], *, clock: Now = _utc_now
    ) -> None:
        self._sessions = sessions
        self._clock = clock

    async def get(self, key: str) -> dict[str, Any] | None:
        """Return the live value, or None. Reads never write: expired rows are filtered."""
        now = self._clock()
        stmt = select(CacheEntry.value).where(
            CacheEntry.key == key,
            or_(CacheEntry.expires_at.is_(None), CacheEntry.expires_at > now),
        )
        async with self._sessions() as session:
            return await session.scalar(stmt)

    async def set(self, key: str, value: dict[str, Any], *, ttl: float | None = None) -> None:
        """Insert or overwrite atomically; an overwrite replaces the value and the TTL."""
        expires_at = None if ttl is None else self._clock() + timedelta(seconds=ttl)
        stmt = insert(CacheEntry).values(key=key, value=value, expires_at=expires_at)
        stmt = stmt.on_conflict_do_update(
            index_elements=[CacheEntry.key],
            set_={'value': stmt.excluded.value, 'expires_at': stmt.excluded.expires_at},
        )
        async with self._sessions() as session, session.begin():
            await session.execute(stmt)

    async def purge_expired(self) -> int:
        """Delete expired rows and return how many were removed."""
        stmt = (
            delete(CacheEntry)
            .where(CacheEntry.expires_at <= self._clock())
            .returning(CacheEntry.key)
        )
        async with self._sessions() as session, session.begin():
            result = await session.scalars(stmt)
            return len(result.all())
