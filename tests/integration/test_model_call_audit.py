from collections.abc import Sequence
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from agentic_ops.db.models import AuditLog
from agentic_ops.db.session import get_sessionmaker
from agentic_ops.gateway.audit import record_model_call, record_model_failure
from agentic_ops.gateway.errors import GatewayExhausted
from agentic_ops.gateway.models import Alias, ModelSpec, Skipped
from agentic_ops.gateway.router import RouteOutcome

pytestmark = pytest.mark.integration

SPEC = ModelSpec(provider='groq', model_id='model-a', supports_tools=True)


@pytest.fixture
def session_factory(async_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return get_sessionmaker(async_engine)


async def _events(factory: async_sessionmaker[AsyncSession], thread_id: str) -> Sequence[AuditLog]:
    async with factory() as session:
        result = await session.scalars(
            select(AuditLog).where(AuditLog.thread_id == thread_id).order_by(AuditLog.id)
        )
        return result.all()


async def test_record_model_call_writes_a_model_call_event(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    thread_id = uuid4().hex
    outcome = RouteOutcome(value='ok', spec=SPEC, skipped=())
    async with session_factory() as session, session.begin():
        await record_model_call(
            session,
            thread_id=thread_id,
            actor='procurement',
            alias=Alias.DEFAULT,
            outcome=outcome,
            estimated_tokens=50,
            latency_ms=12.5,
        )
    (event,) = await _events(session_factory, thread_id)
    assert event.event_type == 'model_call'
    assert event.actor == 'procurement'
    assert event.payload['model_id'] == 'model-a'
    assert event.payload['cache_hit'] is False


async def test_record_model_failure_writes_a_failed_event(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    thread_id = uuid4().hex
    error = GatewayExhausted('all failed', skipped=(Skipped(spec=SPEC, reason='unavailable'),))
    async with session_factory() as session, session.begin():
        await record_model_failure(
            session,
            thread_id=thread_id,
            actor='procurement',
            alias=Alias.DEFAULT,
            error=error,
            estimated_tokens=50,
            latency_ms=3.0,
        )
    (event,) = await _events(session_factory, thread_id)
    assert event.event_type == 'model_call_failed'
    assert event.payload['skipped'][0]['reason'] == 'unavailable'
