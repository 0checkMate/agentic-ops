"""Procurement tools (thin ErpPort wrappers) and the agent-to-specialist adapter."""

import json
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool

from agentic_ops.agents.procurement import (
    MAX_PAGE_SIZE,
    agent_as_specialist,
    build_procurement_tools,
)
from agentic_ops.agents.supervisor import AgentState
from tests.unit.support import DRAFT_ARGS, FakeErp


def tool_by_name(tools: Sequence[BaseTool], name: str) -> BaseTool:
    for candidate in tools:
        if candidate.name == name:
            return candidate
    raise AssertionError(f'tool {name} not found')


async def call(erp: FakeErp, name: str, args: dict[str, Any]) -> str:
    result = await tool_by_name(build_procurement_tools(erp), name).ainvoke(args)
    assert isinstance(result, str)
    return result


def test_tool_names_expose_no_confirmation() -> None:
    names = {tool.name for tool in build_procurement_tools(FakeErp())}
    assert names == {
        'list_products',
        'get_stock_levels',
        'get_vendor',
        'create_draft_po',
        'get_purchase_order',
    }
    assert not any('confirm' in n or 'approve' in n for n in names)


async def test_get_stock_levels_returns_compact_json() -> None:
    out = json.loads(await call(FakeErp(), 'get_stock_levels', {'product_ref': 'P1'}))
    assert out == [
        {
            'product_ref': 'P1',
            'location': 'WH/Stock',
            'quantity': '120.5',
            'reserved_quantity': '0',
        }
    ]


async def test_list_products_clamps_page_size() -> None:
    erp = FakeErp()
    await call(erp, 'list_products', {'limit': 1000})
    assert erp.list_limits == [MAX_PAGE_SIZE]


async def test_not_found_becomes_error_text() -> None:
    out = await call(FakeErp(), 'get_vendor', {'partner_ref': 'V9'})
    assert out.startswith('ERROR: ')


async def test_validation_error_becomes_error_text() -> None:
    out = await call(FakeErp(), 'create_draft_po', {**DRAFT_ARGS, 'partner_ref': 'V9'})
    assert out.startswith('ERROR: ')


async def test_create_draft_po_returns_a_draft() -> None:
    erp = FakeErp()
    out = json.loads(await call(erp, 'create_draft_po', DRAFT_ARGS))
    assert out['state'] == 'draft'
    assert out['ref'] == 'PO0001'
    assert len(erp.orders) == 1


async def test_identical_requests_create_one_draft() -> None:
    erp = FakeErp()
    first = json.loads(await call(erp, 'create_draft_po', DRAFT_ARGS))
    second = json.loads(await call(erp, 'create_draft_po', DRAFT_ARGS))
    assert first['ref'] == second['ref']
    assert len(erp.orders) == 1


async def test_different_requests_create_different_drafts() -> None:
    erp = FakeErp()
    await call(erp, 'create_draft_po', DRAFT_ARGS)
    other = {
        'partner_ref': 'V1',
        'lines': [{'product_ref': 'P1', 'product_qty': '11', 'price_unit': '2.5'}],
    }
    await call(erp, 'create_draft_po', other)
    assert len(erp.orders) == 2


async def test_adapter_returns_only_new_messages() -> None:
    async def fake_agent(payload: dict[str, Any]) -> dict[str, Any]:
        history: list[BaseMessage] = list(payload['messages'])
        trace: list[BaseMessage] = [
            AIMessage(
                content='',
                tool_calls=[
                    {
                        'name': 'get_vendor',
                        'args': {'partner_ref': 'V1'},
                        'id': 'c1',
                        'type': 'tool_call',
                    }
                ],
            ),
            ToolMessage(content='{}', tool_call_id='c1'),
            AIMessage(content='answer'),
        ]
        return {'messages': [*history, *trace]}

    specialist = agent_as_specialist(RunnableLambda(fake_agent))
    state: AgentState = {'messages': [HumanMessage('question')]}
    result = await specialist(state)
    returned = result['messages']
    assert len(returned) == 3
    assert not any(isinstance(m, HumanMessage) for m in returned)
    assert returned[-1].content == 'answer'
