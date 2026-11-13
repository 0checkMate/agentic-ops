"""The read-only SQL role: SELECT on mock_erp only; writes and app tables refused."""

import secrets
from collections.abc import AsyncIterator
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool

from agentic_ops.erp.mock.demo_seed import build_demo_dataset
from agentic_ops.erp.mock.loader import (
    READONLY_ROLE,
    SCHEMA,
    load_dataset,
    provision_readonly_role,
)

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures('migrated_db')]

DATASET = build_demo_dataset()


def _role_engine(test_db_url: str, password: str) -> AsyncEngine:
    url = make_url(test_db_url).set(username=READONLY_ROLE, password=password)
    return create_async_engine(url, poolclass=NullPool)


@pytest.fixture
async def readonly_engine(
    async_engine: AsyncEngine, test_db_url: str
) -> AsyncIterator[AsyncEngine]:
    password = secrets.token_hex(16)
    async with async_engine.begin() as conn:
        await load_dataset(conn, DATASET)
        await provision_readonly_role(conn, password)
    engine = _role_engine(test_db_url, password)
    yield engine
    await engine.dispose()


async def _scalar(engine: AsyncEngine, sql: str) -> Any:
    async with engine.connect() as conn:
        return (await conn.execute(text(sql))).scalar_one()


async def _read_write(conn: AsyncConnection) -> None:
    """Lift the read-only default so a test proves the privilege layer, not the flag."""
    await conn.execute(text('SET TRANSACTION READ WRITE'))


async def test_role_can_read_the_mock_erp_schema(readonly_engine: AsyncEngine) -> None:
    count = await _scalar(readonly_engine, f'SELECT count(*) FROM {SCHEMA}.product')
    assert count == len(DATASET.products)


async def test_role_cannot_write_to_mock_erp(readonly_engine: AsyncEngine) -> None:
    async with readonly_engine.connect() as conn:
        await _read_write(conn)
        with pytest.raises(DBAPIError, match='permission denied'):
            await conn.execute(text(f"INSERT INTO {SCHEMA}.customer (ref, name) VALUES ('X', 'Y')"))


async def test_role_cannot_read_application_tables(readonly_engine: AsyncEngine) -> None:
    async with readonly_engine.connect() as conn:
        await _read_write(conn)
        with pytest.raises(DBAPIError, match='permission denied'):
            await conn.execute(text('SELECT count(*) FROM public.audit_log'))


async def test_role_cannot_create_objects(readonly_engine: AsyncEngine) -> None:
    async with readonly_engine.connect() as conn:
        await _read_write(conn)
        with pytest.raises(DBAPIError, match='permission denied'):
            await conn.execute(text('CREATE TABLE public.scratch (id integer)'))


async def test_read_only_flag_blocks_writes_on_its_own(readonly_engine: AsyncEngine) -> None:
    async with readonly_engine.connect() as conn:
        with pytest.raises(DBAPIError, match='read-only transaction'):
            await conn.execute(text(f"INSERT INTO {SCHEMA}.customer (ref, name) VALUES ('X', 'Y')"))


async def test_role_sessions_are_read_only_by_default(readonly_engine: AsyncEngine) -> None:
    value = await _scalar(readonly_engine, 'SHOW default_transaction_read_only')
    assert value == 'on'


async def test_unsafe_password_is_rejected_before_any_sql(async_engine: AsyncEngine) -> None:
    async with async_engine.connect() as conn:
        with pytest.raises(ValueError):
            await provision_readonly_role(conn, "x'; DROP SCHEMA public; --")


async def test_provisioning_twice_rotates_the_password(
    async_engine: AsyncEngine, test_db_url: str
) -> None:
    second = secrets.token_hex(16)
    async with async_engine.begin() as conn:
        await load_dataset(conn, DATASET)
        await provision_readonly_role(conn, secrets.token_hex(16))
        await provision_readonly_role(conn, second)
    engine = _role_engine(test_db_url, second)
    try:
        assert await _scalar(engine, f'SELECT count(*) FROM {SCHEMA}.customer') > 0
    finally:
        await engine.dispose()
