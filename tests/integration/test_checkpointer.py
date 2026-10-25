from typing import Any, TypedDict
from uuid import uuid4

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from agentic_ops.db.checkpoint import open_checkpointer

pytestmark = pytest.mark.integration


class CounterState(TypedDict):
    count: int


def _increment(state: CounterState) -> CounterState:
    return {'count': state['count'] + 1}


def _build(checkpointer: Any) -> Any:
    builder = StateGraph(CounterState)
    builder.add_node('increment', _increment)
    builder.add_edge(START, 'increment')
    builder.add_edge('increment', END)
    return builder.compile(checkpointer=checkpointer)


def _config() -> RunnableConfig:
    return {'configurable': {'thread_id': f'test-{uuid4()}'}}


async def test_state_survives_a_new_checkpointer(test_db_url: str) -> None:
    config = _config()
    async with open_checkpointer(test_db_url) as saver:
        result = await _build(saver).ainvoke({'count': 0}, config)
    assert result['count'] == 1
    async with open_checkpointer(test_db_url) as saver:
        snapshot = await _build(saver).aget_state(config)
    assert snapshot.values['count'] == 1


async def test_threads_are_isolated(test_db_url: str) -> None:
    first, second = _config(), _config()
    async with open_checkpointer(test_db_url) as saver:
        graph = _build(saver)
        await graph.ainvoke({'count': 10}, first)
        await graph.ainvoke({'count': 0}, second)
        one = await graph.aget_state(first)
        two = await graph.aget_state(second)
    assert one.values['count'] == 11
    assert two.values['count'] == 1


async def test_history_records_each_step(test_db_url: str) -> None:
    config = _config()
    async with open_checkpointer(test_db_url) as saver:
        graph = _build(saver)
        await graph.ainvoke({'count': 0}, config)
        history = [state async for state in graph.aget_state_history(config)]
    assert len(history) >= 2
