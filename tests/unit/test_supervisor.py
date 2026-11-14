"""Supervisor routing: handoff by tool call, direct answer, unknown agent, persistence."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from agentic_ops.agents.supervisor import UNROUTABLE_REPLY, build_supervisor_graph
from tests.unit.support import ScriptedChatModel, recording_specialist, route_call

CONFIG: RunnableConfig = {'configurable': {'thread_id': 'thread-1'}}


async def test_routes_to_specialist_and_ends_turn() -> None:
    calls: list[str] = []
    model = ScriptedChatModel(script=[route_call('procurement')])
    graph = build_supervisor_graph(
        model,
        {'procurement': recording_specialist('procurement', calls)},
        checkpointer=InMemorySaver(),
    )
    result = await graph.ainvoke({'messages': [HumanMessage('Which supplier for flour?')]}, CONFIG)
    assert calls == ['procurement']
    assert result['messages'][-1].content == 'procurement done'
    assert model.cursor == 1  # one LLM call: specialists end the turn


async def test_answers_directly_without_tool_call() -> None:
    calls: list[str] = []
    model = ScriptedChatModel(script=[AIMessage(content='Hello, how can I help?')])
    graph = build_supervisor_graph(
        model,
        {'procurement': recording_specialist('procurement', calls)},
        checkpointer=InMemorySaver(),
    )
    result = await graph.ainvoke({'messages': [HumanMessage('hi')]}, CONFIG)
    assert calls == []
    assert result['messages'][-1].content == 'Hello, how can I help?'


async def test_unknown_agent_is_not_followed() -> None:
    calls: list[str] = []
    model = ScriptedChatModel(script=[route_call('payroll')])
    graph = build_supervisor_graph(
        model,
        {'procurement': recording_specialist('procurement', calls)},
        checkpointer=InMemorySaver(),
    )
    result = await graph.ainvoke({'messages': [HumanMessage('pay everyone')]}, CONFIG)
    assert calls == []
    assert result['messages'][-1].content == UNROUTABLE_REPLY


async def test_routing_message_is_not_persisted() -> None:
    model = ScriptedChatModel(script=[route_call('procurement')])
    graph = build_supervisor_graph(
        model,
        {'procurement': recording_specialist('procurement', [])},
        checkpointer=InMemorySaver(),
    )
    result = await graph.ainvoke({'messages': [HumanMessage('reorder flour')]}, CONFIG)
    assert not any(isinstance(m, AIMessage) and m.tool_calls for m in result['messages'])


async def test_prompt_lists_every_specialist() -> None:
    model = ScriptedChatModel(script=[AIMessage(content='ok')])
    specialists = {
        'procurement': recording_specialist('procurement', []),
        'quality': recording_specialist('quality', []),
    }
    graph = build_supervisor_graph(model, specialists, checkpointer=InMemorySaver())
    await graph.ainvoke({'messages': [HumanMessage('hi')]}, CONFIG)
    first = model.seen[0][0]
    assert isinstance(first, SystemMessage)
    assert 'procurement' in str(first.content)
    assert 'quality' in str(first.content)


async def test_thread_history_persists_across_turns() -> None:
    model = ScriptedChatModel(script=[route_call('procurement'), AIMessage(content='ok')])
    graph = build_supervisor_graph(
        model,
        {'procurement': recording_specialist('procurement', [])},
        checkpointer=InMemorySaver(),
    )
    await graph.ainvoke({'messages': [HumanMessage('first question')]}, CONFIG)
    await graph.ainvoke({'messages': [HumanMessage('second question')]}, CONFIG)
    assert any(isinstance(m, HumanMessage) and m.content == 'first question' for m in model.seen[1])
