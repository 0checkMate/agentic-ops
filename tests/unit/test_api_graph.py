"""build_chat_graph wires supervisor, procurement, checkpointer and audit together."""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from agentic_ops.agents.audit import AuditEvent
from agentic_ops.api.main import build_chat_graph, create_live_app
from tests.unit.support import FakeErp, ScriptedChatModel, route_call, tool_call_message

THREAD: RunnableConfig = {'configurable': {'thread_id': 't-1'}}


async def test_turn_is_routed_and_audited_in_order() -> None:
    events: list[AuditEvent] = []

    async def sink(event: AuditEvent) -> None:
        events.append(event)

    model = ScriptedChatModel(
        script=[
            route_call('procurement'),
            tool_call_message('get_vendor', {'partner_ref': 'V1'}, 'call_1'),
            AIMessage(content='Unga Ltd is a supplier.'),
        ]
    )
    graph = build_chat_graph(model, FakeErp(), checkpointer=InMemorySaver(), audit=sink)
    result = await graph.ainvoke({'messages': [HumanMessage('Who is V1?')]}, THREAD)
    last = result['messages'][-1]
    assert isinstance(last, AIMessage)
    assert last.content == 'Unga Ltd is a supplier.'
    assert [e.event_type for e in events] == ['route_decision', 'tool_call', 'tool_result']
    assert {e.thread_id for e in events} == {'t-1'}


async def test_reply_carries_the_specialist_name() -> None:
    async def sink(event: AuditEvent) -> None:
        return None

    model = ScriptedChatModel(script=[route_call('procurement'), AIMessage(content='Done.')])
    graph = build_chat_graph(model, FakeErp(), checkpointer=InMemorySaver(), audit=sink)
    result = await graph.ainvoke({'messages': [HumanMessage('hi')]}, THREAD)
    assert result['messages'][-1].name == 'procurement'


async def test_second_turn_sees_the_first_turn() -> None:
    async def sink(event: AuditEvent) -> None:
        return None

    model = ScriptedChatModel(
        script=[
            route_call('procurement'),
            AIMessage(content='First.'),
            route_call('procurement'),
            AIMessage(content='Second.'),
        ]
    )
    graph = build_chat_graph(model, FakeErp(), checkpointer=InMemorySaver(), audit=sink)
    await graph.ainvoke({'messages': [HumanMessage('first')]}, THREAD)
    await graph.ainvoke({'messages': [HumanMessage('second')]}, THREAD)
    assert any(isinstance(m, HumanMessage) and m.content == 'first' for m in model.seen[-1])


def test_live_app_exposes_probes_and_chat() -> None:
    paths = set(create_live_app().openapi()['paths'])
    assert {'/health', '/ready', '/chat'} <= paths
