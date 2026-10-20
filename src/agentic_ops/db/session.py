from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from agentic_ops.config import get_settings


def make_engine(url: str, *, null_pool: bool = False) -> AsyncEngine:
    """Build an async engine tuned for Neon's idle-suspend behaviour.

    pool_pre_ping detects sockets killed by a compute suspend; pool_recycle retires
    connections before Neon's 5-minute idle window closes them.
    """
    if null_pool:
        return create_async_engine(url, poolclass=NullPool)
    return create_async_engine(url, pool_pre_ping=True, pool_recycle=240)


@lru_cache
def get_engine() -> AsyncEngine:
    """Process-wide engine built from application settings."""
    return make_engine(get_settings().sqlalchemy_url)


def get_sessionmaker(engine: AsyncEngine | None = None) -> async_sessionmaker[AsyncSession]:
    """Session factory; expire_on_commit=False avoids implicit lazy loads in async code."""
    return async_sessionmaker(engine or get_engine(), expire_on_commit=False)
