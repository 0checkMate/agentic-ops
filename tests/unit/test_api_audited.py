"""AuditedGraph attaches the audit handler to every run and touches nothing else."""

from typing import Any

import pytest
from langchain_core.callbacks import BaseCallbackHandler, CallbackManager
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agentic_ops.agents.audit import AuditCallbackHandler, AuditEvent
from agentic_ops.agents.supervisor import AgentState
from agentic_ops.api.audited import AuditedGraph


async def discard(event: AuditEvent) -> None:
    return None


class RecordingGraph:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[tuple[AgentState, RunnableConfig]] = []

    async def ainvoke(self, input: AgentState, config: RunnableConfig, /) -> Any:
        self.calls.append((input, config))
        return self.result


def ok_result() -> dict[str, Any]:
    return {'messages': [AIMessage(content='ok')]}


async def test_handler_is_attached_and_other_config_is_preserved() -> None:
    inner = RecordingGraph(ok_result())
    handler = AuditCallbackHandler(discard)
    config: RunnableConfig = {'configurable': {'thread_id': 't-1'}}
    await AuditedGraph(inner, handler).ainvoke({'messages': [HumanMessage('hi')]}, config)
    _, seen = inner.calls[0]
    assert seen.get('callbacks') == [handler]
    assert seen.get('configurable') == {'thread_id': 't-1'}
    assert 'callbacks' not in config


async def test_existing_callbacks_are_kept() -> None:
    inner = RecordingGraph(ok_result())
    handler = AuditCallbackHandler(discard)
    other = BaseCallbackHandler()
    config: RunnableConfig = {'callbacks': [other]}
    await AuditedGraph(inner, handler).ainvoke({'messages': []}, config)
    assert inner.calls[0][1].get('callbacks') == [other, handler]


async def test_messages_are_forwarded_unchanged() -> None:
    inner = RecordingGraph(ok_result())
    message = HumanMessage('Order 10 kg of sugar')
    await AuditedGraph(inner, AuditCallbackHandler(discard)).ainvoke({'messages': [message]}, {})
    assert inner.calls[0][0]['messages'] == [message]


async def test_result_is_returned() -> None:
    result = ok_result()
    graph = AuditedGraph(RecordingGraph(result), AuditCallbackHandler(discard))
    assert await graph.ainvoke({'messages': []}, {}) == result


async def test_non_dict_result_is_rejected() -> None:
    graph = AuditedGraph(RecordingGraph('not a dict'), AuditCallbackHandler(discard))
    with pytest.raises(TypeError):
        await graph.ainvoke({'messages': []}, {})


async def test_callback_manager_in_config_is_refused() -> None:
    graph = AuditedGraph(RecordingGraph(ok_result()), AuditCallbackHandler(discard))
    config: RunnableConfig = {'callbacks': CallbackManager(handlers=[])}
    with pytest.raises(TypeError):
        await graph.ainvoke({'messages': []}, config)
