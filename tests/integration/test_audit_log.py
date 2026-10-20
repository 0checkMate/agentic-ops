import uuid
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.exc import StatementError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from agentic_ops.db.audit import record_event
from agentic_ops.db.models import AuditLog
from agentic_ops.db.session import get_sessionmaker

pytestmark = pytest.mark.integration

SessionFactory = async_sessionmaker[AsyncSession]


@pytest.fixture
def session_factory(async_engine: AsyncEngine) -> SessionFactory:
    return get_sessionmaker(async_engine)


def _new_thread_id() -> str:
    return uuid.uuid4().hex


async def _fetch(session_factory: SessionFactory, thread_id: str) -> list[AuditLog]:
    stmt = select(AuditLog).where(AuditLog.thread_id == thread_id).order_by(AuditLog.id)
    async with session_factory() as session:
        return list((await session.scalars(stmt)).all())


async def test_event_round_trip(session_factory: SessionFactory) -> None:
    thread_id = _new_thread_id()
    async with session_factory() as session:
        event = await record_event(
            session,
            thread_id=thread_id,
            event_type='tool_call',
            actor='procurement_agent',
            payload={'tool': 'get_stock_levels'},
        )
        # Server default must arrive via RETURNING: no lazy load, no MissingGreenlet.
        assert event.id is not None
        assert event.created_at is not None
        await session.commit()

    rows = await _fetch(session_factory, thread_id)
    assert len(rows) == 1
    row = rows[0]
    assert row.event_type == 'tool_call'
    assert row.actor == 'procurement_agent'
    assert row.payload == {'tool': 'get_stock_levels'}
    assert row.created_at.tzinfo is not None


async def test_default_payload_is_empty_object(session_factory: SessionFactory) -> None:
    thread_id = _new_thread_id()
    async with session_factory() as session:
        await record_event(session, thread_id=thread_id, event_type='note', actor='system')
        await session.commit()

    rows = await _fetch(session_factory, thread_id)
    assert rows[0].payload == {}


async def test_nested_payload_round_trip(session_factory: SessionFactory) -> None:
    thread_id = _new_thread_id()
    payload = {
        'lines': [{'sku': 'JUICE-001', 'qty': '12.500'}, {'sku': 'JUICE-002', 'qty': '3'}],
        'meta': {'approved': False, 'ratio': 0.5, 'note': None},
    }
    async with session_factory() as session:
        await record_event(
            session,
            thread_id=thread_id,
            event_type='draft_po',
            actor='procurement_agent',
            payload=payload,
        )
        await session.commit()

    rows = await _fetch(session_factory, thread_id)
    assert rows[0].payload == payload


async def test_thread_ordering_and_isolation(session_factory: SessionFactory) -> None:
    thread_a = _new_thread_id()
    thread_b = _new_thread_id()
    async with session_factory() as session:
        await record_event(session, thread_id=thread_a, event_type='first', actor='system')
        await record_event(session, thread_id=thread_b, event_type='other', actor='system')
        await record_event(session, thread_id=thread_a, event_type='second', actor='system')
        await record_event(session, thread_id=thread_a, event_type='third', actor='system')
        await session.commit()

    rows_a = await _fetch(session_factory, thread_a)
    rows_b = await _fetch(session_factory, thread_b)
    assert [row.event_type for row in rows_a] == ['first', 'second', 'third']
    assert [row.event_type for row in rows_b] == ['other']


async def test_rollback_leaves_no_row(session_factory: SessionFactory) -> None:
    thread_id = _new_thread_id()
    async with session_factory() as session:
        await record_event(session, thread_id=thread_id, event_type='doomed', actor='system')
        await session.rollback()

    assert await _fetch(session_factory, thread_id) == []


async def test_non_json_payload_is_rejected(session_factory: SessionFactory) -> None:
    thread_id = _new_thread_id()
    async with session_factory() as session:
        with pytest.raises((StatementError, TypeError)):
            await record_event(
                session,
                thread_id=thread_id,
                event_type='bad_payload',
                actor='system',
                payload={'qty': Decimal('1.5')},
            )
        await session.rollback()

    assert await _fetch(session_factory, thread_id) == []
