"""LazyGraph holds the graph that the lifespan builds after startup."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from agentic_ops.api.app import create_app
from agentic_ops.api.lazy import GraphNotReady, LazyGraph


class EchoGraph:
    async def ainvoke(self, input: dict[str, Any], config: RunnableConfig, /) -> dict[str, Any]:
        return {'messages': [*input['messages'], AIMessage(content='ready')]}


async def no_db() -> bool:
    return True


async def test_raises_before_set() -> None:
    lazy = LazyGraph()
    with pytest.raises(GraphNotReady):
        await lazy.ainvoke({'messages': []}, {})


async def test_delegates_after_set() -> None:
    lazy = LazyGraph()
    lazy.set(EchoGraph())
    result = await lazy.ainvoke({'messages': []}, {})
    assert result['messages'][-1].content == 'ready'


async def test_clear_returns_to_not_ready() -> None:
    lazy = LazyGraph()
    lazy.set(EchoGraph())
    lazy.clear()
    with pytest.raises(GraphNotReady):
        await lazy.ainvoke({'messages': []}, {})


async def test_chat_returns_502_before_startup_completes() -> None:
    app = create_app(no_db, graph=LazyGraph())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
        response = await client.post('/chat', json={'thread_id': 't-1', 'message': 'hi'})
    assert response.status_code == 502


async def test_lifespan_runs_around_serving() -> None:
    events: list[str] = []

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        events.append('start')
        yield
        events.append('stop')

    app = create_app(no_db, lifespan=lifespan)
    async with app.router.lifespan_context(app):
        events.append('serving')
    assert events == ['start', 'serving', 'stop']
