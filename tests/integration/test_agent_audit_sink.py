"""PostgresAuditSink: each event commits in its own transaction, ordered by id."""

from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from agentic_ops.agents.audit import AuditEvent, PostgresAuditSink
from agentic_ops.db.models import AuditLog
from agentic_ops.db.session import get_sessionmaker

pytestmark = pytest.mark.integration


@pytest.fixture
def session_factory(async_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return get_sessionmaker(async_engine)


async def test_events_are_committed_and_ordered(
    session_factory: async_sessionmaker[AsyncSession], migrated_db: object
) -> None:
    thread_id = f'sink-{uuid4()}'
    sink = PostgresAuditSink(session_factory)
    await sink(
        AuditEvent(
            thread_id=thread_id,
            event_type='route_decision',
            actor='supervisor',
            payload={'outcome': 'routed', 'agent': 'procurement'},
        )
    )
    await sink(
        AuditEvent(
            thread_id=thread_id,
            event_type='tool_call',
            actor='agent',
            payload={'tool': 'get_vendor', 'args': {}},
        )
    )
    async with session_factory() as session:  # a different session: proves the commit
        rows = (
            await session.scalars(
                select(AuditLog).where(AuditLog.thread_id == thread_id).order_by(AuditLog.id)
            )
        ).all()
    assert [r.event_type for r in rows] == ['route_decision', 'tool_call']
    assert rows[0].payload == {'outcome': 'routed', 'agent': 'procurement'}
