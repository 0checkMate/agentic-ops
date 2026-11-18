"""Agent audit: route and tool events, recorded per thread ID."""

import json
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agentic_ops.db.audit import record_event

PREVIEW_CHARS = 500
TOOL_ACTOR = 'agent'


class AuditEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    thread_id: str
    event_type: str
    actor: str
    payload: dict[str, Any]


AuditSink = Callable[[AuditEvent], Awaitable[None]]


def thread_id_from_config(config: RunnableConfig) -> str:
    """Return the thread ID of a run, refusing runs that have none."""
    value = config.get('configurable', {}).get('thread_id')
    if not isinstance(value, str) or not value:
        raise ValueError('thread_id is required in config["configurable"] for audited runs')
    return value


class PostgresAuditSink:
    """Writes each event in its own short transaction (survives a mid-turn crash)."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def __call__(self, event: AuditEvent) -> None:
        async with self._session_factory() as session, session.begin():
            await record_event(
                session,
                thread_id=event.thread_id,
                event_type=event.event_type,
                actor=event.actor,
                payload=event.payload,
            )


class AuditCallbackHandler(AsyncCallbackHandler):
    """Records every tool run of any agent, so new specialists need no audit code."""

    raise_error = True  # fail closed: a dropped audit event aborts the turn
    run_inline = True  # keep events in execution order

    def __init__(self, sink: AuditSink) -> None:
        self._sink = sink
        self._runs: dict[UUID, tuple[str, str]] = {}

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        thread_id = (metadata or {}).get('thread_id')
        if not isinstance(thread_id, str) or not thread_id:
            raise ValueError('thread_id missing from tool run metadata')
        name = kwargs.get('name') or serialized.get('name')
        tool = name if isinstance(name, str) else 'unknown'
        self._runs[run_id] = (thread_id, tool)
        args = json.loads(json.dumps(inputs or {}, default=str))
        await self._sink(
            AuditEvent(
                thread_id=thread_id,
                event_type='tool_call',
                actor=TOOL_ACTOR,
                payload={'tool': tool, 'args': args},
            )
        )

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        thread_id, tool = self._runs.pop(run_id)
        text = str(output.content if isinstance(output, ToolMessage) else output)
        await self._sink(
            AuditEvent(
                thread_id=thread_id,
                event_type='tool_result',
                actor=TOOL_ACTOR,
                payload={
                    'tool': tool,
                    'ok': not text.startswith('ERROR: '),
                    'output_preview': text[:PREVIEW_CHARS],
                },
            )
        )
