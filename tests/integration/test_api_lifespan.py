"""The real lifespan on Postgres: state persists across turns and audit rows are written."""

from uuid import uuid4

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from agentic_ops.api.app import create_app
from agentic_ops.api.lazy import LazyGraph
from agentic_ops.api.main import live_lifespan
from agentic_ops.config import Settings
from agentic_ops.db.session import get_sessionmaker
from tests.unit.support import ScriptedChatModel, route_call, tool_call_message

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures('migrated_db')]


async def no_db() -> bool:
    return True


async def test_state_persists_and_every_step_is_audited(
    test_db_url: str, async_engine: AsyncEngine
) -> None:
    model = ScriptedChatModel(
        script=[
            route_call('procurement'),
            tool_call_message('get_vendor', {'partner_ref': 'V-001'}, 'call_1'),
            AIMessage(content='Vendor found.'),
            route_call('procurement'),
            AIMessage(content='Second turn.'),
        ]
    )
    settings = Settings(database_url=SecretStr(test_db_url))
    lazy = LazyGraph()
    lifespan = live_lifespan(settings, lazy, get_sessionmaker(async_engine), model=model)
    app = create_app(no_db, graph=lazy, lifespan=lifespan)
    thread_id = f'it-{uuid4().hex}'

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
            first = await client.post('/chat', json={'thread_id': thread_id, 'message': 'first'})
            second = await client.post('/chat', json={'thread_id': thread_id, 'message': 'next'})

    assert first.status_code == 200
    assert first.json()['reply'] == 'Vendor found.'
    assert second.status_code == 200
    assert any(isinstance(m, HumanMessage) and m.content == 'first' for m in model.seen[-1])
    async with async_engine.connect() as connection:
        result = await connection.execute(
            text('SELECT event_type FROM audit_log WHERE thread_id = :t ORDER BY id'),
            {'t': thread_id},
        )
        event_types = [row[0] for row in result]
    assert event_types == ['route_decision', 'tool_call', 'tool_result', 'route_decision']
