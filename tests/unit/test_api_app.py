"""HTTP contract of the app factory, exercised in memory with a fake graph."""

from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agentic_ops.api.app import create_app
from agentic_ops.api.chat import ChatGraph


class FakeGraph:
    """Records every invocation and either replies or raises."""

    def __init__(self, reply: BaseMessage, error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.calls: list[tuple[dict[str, Any], RunnableConfig]] = []

    async def ainvoke(self, input: dict[str, Any], config: RunnableConfig, /) -> dict[str, Any]:
        self.calls.append((input, config))
        if self.error is not None:
            raise self.error
        return {'messages': [*input['messages'], self.reply]}


async def no_db() -> bool:
    return True


def make_client(graph: ChatGraph | None) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=create_app(no_db, graph=graph))
    return httpx.AsyncClient(transport=transport, base_url='http://test')


async def test_health_is_ok_without_touching_the_graph() -> None:
    graph = FakeGraph(AIMessage(content='unused'))
    async with make_client(graph) as client:
        response = await client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}
    assert graph.calls == []


async def test_chat_returns_reply_agent_and_thread_id() -> None:
    graph = FakeGraph(AIMessage(content='Draft PO created.', name='procurement'))
    async with make_client(graph) as client:
        response = await client.post(
            '/chat', json={'thread_id': 't-1', 'message': 'Order 10 kg of sugar'}
        )
    assert response.status_code == 200
    assert response.json() == {
        'thread_id': 't-1',
        'reply': 'Draft PO created.',
        'agent': 'procurement',
    }
    sent, _ = graph.calls[0]
    assert isinstance(sent['messages'][0], HumanMessage)
    assert sent['messages'][0].content == 'Order 10 kg of sugar'


async def test_direct_answer_has_no_agent() -> None:
    graph = FakeGraph(AIMessage(content='Hello.'))
    async with make_client(graph) as client:
        response = await client.post('/chat', json={'thread_id': 't-1', 'message': 'hi'})
    assert response.json()['agent'] is None


async def test_block_content_is_flattened_to_text() -> None:
    graph = FakeGraph(AIMessage(content=[{'type': 'text', 'text': 'Hello'}, ' world']))
    async with make_client(graph) as client:
        response = await client.post('/chat', json={'thread_id': 't-1', 'message': 'hi'})
    assert response.json()['reply'] == 'Hello world'


async def test_same_thread_id_reaches_the_graph_on_every_turn() -> None:
    graph = FakeGraph(AIMessage(content='ok'))
    async with make_client(graph) as client:
        await client.post('/chat', json={'thread_id': 'abc', 'message': 'one'})
        await client.post('/chat', json={'thread_id': 'abc', 'message': 'two'})
    thread_ids = [config['configurable']['thread_id'] for _, config in graph.calls]
    assert thread_ids == ['abc', 'abc']


@pytest.mark.parametrize(
    'payload',
    [
        {'message': 'hi'},
        {'thread_id': '', 'message': 'hi'},
        {'thread_id': 'has space', 'message': 'hi'},
        {'thread_id': 'x' * 129, 'message': 'hi'},
        {'thread_id': 't-1', 'message': ''},
        {'thread_id': 't-1'},
        {'thread_id': 't-1', 'message': 'hi', 'extra': 'nope'},
    ],
)
async def test_invalid_body_is_rejected_before_the_graph(payload: dict[str, Any]) -> None:
    graph = FakeGraph(AIMessage(content='unused'))
    async with make_client(graph) as client:
        response = await client.post('/chat', json=payload)
    assert response.status_code == 422
    assert graph.calls == []


async def test_graph_failure_returns_502_without_leaking_detail() -> None:
    graph = FakeGraph(AIMessage(content='unused'), error=RuntimeError('secret-db-password'))
    async with make_client(graph) as client:
        response = await client.post('/chat', json={'thread_id': 't-1', 'message': 'hi'})
    assert response.status_code == 502
    assert 'secret-db-password' not in response.text


async def test_non_assistant_final_message_returns_502() -> None:
    graph = FakeGraph(HumanMessage(content='echo'))
    async with make_client(graph) as client:
        response = await client.post('/chat', json={'thread_id': 't-1', 'message': 'hi'})
    assert response.status_code == 502


async def test_chat_route_is_absent_without_a_graph() -> None:
    async with make_client(None) as client:
        response = await client.post('/chat', json={'thread_id': 't-1', 'message': 'hi'})
    assert response.status_code == 404
