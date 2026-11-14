"""End to end: supervisor routes, create_agent runs the tool loop, tools hit FakeErp."""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from agentic_ops.agents.procurement import agent_as_specialist, build_procurement_agent
from agentic_ops.agents.supervisor import AgentState, build_supervisor_graph
from agentic_ops.erp.models import PurchaseOrderState
from tests.unit.support import (
    DRAFT_ARGS,
    FakeErp,
    ScriptedChatModel,
    route_call,
    tool_call_message,
    unpaired_tool_call_ids,
)

CONFIG: RunnableConfig = {'configurable': {'thread_id': 'flow-1'}}


def build_graph(
    model: ScriptedChatModel, erp: FakeErp
) -> CompiledStateGraph[AgentState, Any, AgentState, AgentState]:
    agent = build_procurement_agent(model, erp)
    return build_supervisor_graph(
        model, {'procurement': agent_as_specialist(agent)}, checkpointer=InMemorySaver()
    )


async def test_full_turn_creates_one_draft() -> None:
    erp = FakeErp()
    model = ScriptedChatModel(
        script=[
            route_call('procurement'),
            tool_call_message('create_draft_po', DRAFT_ARGS, 'c1'),
            AIMessage(content='Draft PO0001 created for confirmation.'),
        ]
    )
    result = await build_graph(model, erp).ainvoke(
        {'messages': [HumanMessage('Order 10 kg flour from Unga')]}, CONFIG
    )
    assert result['messages'][-1].content == 'Draft PO0001 created for confirmation.'
    assert len(erp.orders) == 1
    assert next(iter(erp.orders.values())).state == PurchaseOrderState.DRAFT
    assert model.cursor == 3  # 1 routing call + 2 agent calls


async def test_repeated_model_call_creates_no_duplicate() -> None:
    erp = FakeErp()
    model = ScriptedChatModel(
        script=[
            route_call('procurement'),
            tool_call_message('create_draft_po', DRAFT_ARGS, 'c1'),
            tool_call_message('create_draft_po', DRAFT_ARGS, 'c2'),
            AIMessage(content='done'),
        ]
    )
    await build_graph(model, erp).ainvoke({'messages': [HumanMessage('order flour')]}, CONFIG)
    assert len(erp.orders) == 1


async def test_tool_calls_stay_paired_across_turns() -> None:
    erp = FakeErp()
    model = ScriptedChatModel(
        script=[
            route_call('procurement'),
            tool_call_message('get_vendor', {'partner_ref': 'V1'}, 'c1'),
            AIMessage(content='Unga Ltd is a supplier.'),
            AIMessage(content='Anything else?'),
        ]
    )
    graph = build_graph(model, erp)
    first = await graph.ainvoke({'messages': [HumanMessage('who is V1?')]}, CONFIG)
    assert unpaired_tool_call_ids(first['messages']) == set()
    second = await graph.ainvoke({'messages': [HumanMessage('thanks')]}, CONFIG)
    assert unpaired_tool_call_ids(second['messages']) == set()
    assert all(unpaired_tool_call_ids(seen) == set() for seen in model.seen)


async def test_erp_error_reaches_the_model_as_text() -> None:
    erp = FakeErp()
    model = ScriptedChatModel(
        script=[
            route_call('procurement'),
            tool_call_message('create_draft_po', {**DRAFT_ARGS, 'partner_ref': 'V9'}, 'c1'),
            AIMessage(content='That supplier does not exist.'),
        ]
    )
    result = await build_graph(model, erp).ainvoke(
        {'messages': [HumanMessage('order from V9')]}, CONFIG
    )
    tool_messages = [m for m in result['messages'] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    content = tool_messages[0].content
    assert isinstance(content, str)
    assert content.startswith('ERROR: ')
    assert erp.orders == {}
