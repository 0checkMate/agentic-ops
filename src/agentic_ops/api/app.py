import asyncio
import logging
from collections.abc import Awaitable, Callable

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agentic_ops.db.session import get_sessionmaker

logger = logging.getLogger(__name__)

DbCheck = Callable[[], Awaitable[bool]]

_READY_TIMEOUT_SECONDS = 15.0


def sql_db_check(factory: async_sessionmaker[AsyncSession]) -> DbCheck:
    """Readiness probe: the database answers a trivial query."""

    async def check() -> bool:
        async with factory() as session:
            await session.execute(text('SELECT 1'))
        return True

    return check


def create_app(db_check: DbCheck) -> FastAPI:
    """Application factory; the database probe is injected so tests need no database."""
    app = FastAPI(title='agentic-ops')

    @app.get('/health')
    async def health() -> dict[str, str]:
        return {'status': 'ok'}

    @app.get('/ready')
    async def ready() -> JSONResponse:
        try:
            async with asyncio.timeout(_READY_TIMEOUT_SECONDS):
                healthy = await db_check()
        except Exception as exc:
            logger.warning('Readiness check failed (%s)', type(exc).__name__)
            healthy = False
        if healthy:
            return JSONResponse({'status': 'ready'})
        return JSONResponse({'status': 'unavailable'}, status_code=503)

    return app


def create_default_app() -> FastAPI:
    """Production wiring: the real session factory. Run with uvicorn's --factory flag."""
    return create_app(sql_db_check(get_sessionmaker()))
