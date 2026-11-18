"""Agent audit: route decisions and tool events are recorded by thread ID."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from agentic_ops.agents.audit import AuditCallbackHandler, AuditEvent, thread_id_from_config
from agentic_ops.agents.procurement import agent_as_specialist, build_procurement_agent
from agentic_ops.agents.supervisor import AgentState, build_supervisor_graph
from tests.unit.support import (
    DRAFT_ARGS,
    FakeErp,
    ScriptedChatModel,
    recording_specialist,
    route_call,
    tool_call_message,
)

THREAD = 'audit-1'


class ListSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def __call__(self, event: AuditEvent) -> None:
        self.events.append(event)


class ToolFailingSink(ListSink):
    async def __call__(self, event: AuditEvent) -> None:
        if event.event_type == 'tool_call':
            raise RuntimeError('db down')
        await super().__call__(event)


def config_for(sink: ListSink) -> RunnableConfig:
    return {'configurable': {'thread_id': THREAD}, 'callbacks': [AuditCallbackHandler(sink)]}


def full_graph(
    model: ScriptedChatModel, sink: ListSink
) -> CompiledStateGraph[AgentState, Any, AgentState, AgentState]:
    agent = build_procurement_agent(model, FakeErp())
    return build_supervisor_graph(
        model,
        {'procurement': agent_as_specialist(agent)},
        checkpointer=InMemorySaver(),
        audit=sink,
    )


@pytest.mark.parametrize(
    ('reply', 'payload'),
    [
        (route_call('procurement'), {'outcome': 'routed', 'agent': 'procurement'}),
        (AIMessage(content='hello'), {'outcome': 'direct', 'agent': None}),
        (route_call('payroll'), {'outcome': 'unroutable', 'agent': 'payroll'}),
    ],
)
async def test_route_decision_is_recorded(reply: AIMessage, payload: dict[str, Any]) -> None:
    sink = ListSink()
    graph = build_supervisor_graph(
        ScriptedChatModel(script=[reply]),
        {'procurement': recording_specialist('procurement', [])},
        checkpointer=InMemorySaver(),
        audit=sink,
    )
    await graph.ainvoke({'messages': [HumanMessage('x')]}, config_for(sink))
    assert [(e.event_type, e.thread_id, e.actor) for e in sink.events] == [
        ('route_decision', THREAD, 'supervisor')
    ]
    assert sink.events[0].payload == payload


async def test_tool_events_follow_the_route_decision_in_order() -> None:
    sink = ListSink()
    model = ScriptedChatModel(
        script=[
            route_call('procurement'),
            tool_call_message('get_vendor', {'partner_ref': 'V1'}, 'c1'),
            AIMessage(content='ok'),
        ]
    )
    await full_graph(model, sink).ainvoke({'messages': [HumanMessage('who?')]}, config_for(sink))
    assert [e.event_type for e in sink.events] == ['route_decision', 'tool_call', 'tool_result']
    assert {e.thread_id for e in sink.events} == {THREAD}
    assert sink.events[1].payload == {'tool': 'get_vendor', 'args': {'partner_ref': 'V1'}}
    result = sink.events[2].payload
    assert result['tool'] == 'get_vendor'
    assert result['ok'] is True
    assert str(result['output_preview']).startswith('{')


async def test_erp_error_is_flagged_not_ok() -> None:
    sink = ListSink()
    model = ScriptedChatModel(
        script=[
            route_call('procurement'),
            tool_call_message('create_draft_po', {**DRAFT_ARGS, 'partner_ref': 'V9'}, 'c1'),
            AIMessage(content='unknown supplier'),
        ]
    )
    await full_graph(model, sink).ainvoke({'messages': [HumanMessage('order')]}, config_for(sink))
    result = sink.events[-1].payload
    assert result['ok'] is False
    assert str(result['output_preview']).startswith('ERROR: ')


async def test_sink_failure_aborts_the_turn() -> None:
    sink = ToolFailingSink()
    model = ScriptedChatModel(
        script=[
            route_call('procurement'),
            tool_call_message('get_vendor', {'partner_ref': 'V1'}, 'c1'),
            AIMessage(content='ok'),
        ]
    )
    with pytest.raises(RuntimeError, match='db down'):
        await full_graph(model, sink).ainvoke(
            {'messages': [HumanMessage('who?')]}, config_for(sink)
        )


def test_missing_thread_id_is_refused() -> None:
    with pytest.raises(ValueError, match='thread_id'):
        thread_id_from_config({})
