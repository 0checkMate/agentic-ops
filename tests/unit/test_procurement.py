"""Procurement tools (thin ErpPort wrappers) and the agent-to-specialist adapter."""

import json
from collections.abc import Sequence
from decimal import Decimal
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
from agentic_ops.erp.errors import ErpNotFound, ErpValidationError
from agentic_ops.erp.models import (
    NewPurchaseOrder,
    Partner,
    Product,
    PurchaseOrder,
    PurchaseOrderState,
    StockLevel,
)


class FakeErp:
    """Minimal in-memory ErpPort: one product, one supplier, idempotent drafts."""

    def __init__(self) -> None:
        self.orders: dict[str, PurchaseOrder] = {}
        self.list_limits: list[int] = []

    async def list_products(self, *, limit: int = 100, offset: int = 0) -> Sequence[Product]:
        self.list_limits.append(limit)
        return [Product(ref='P1', name='Wheat flour', uom='kg')]

    async def get_stock_levels(self, product_ref: str | None = None) -> Sequence[StockLevel]:
        if product_ref not in (None, 'P1'):
            raise ErpNotFound(f'product {product_ref}')
        return [StockLevel(product_ref='P1', location='WH/Stock', quantity=Decimal('120.5'))]

    async def get_vendor(self, partner_ref: str) -> Partner:
        if partner_ref != 'V1':
            raise ErpNotFound(f'partner {partner_ref}')
        return Partner(ref='V1', name='Unga Ltd', is_supplier=True)

    async def create_draft_po(self, order: NewPurchaseOrder) -> PurchaseOrder:
        if order.partner_ref != 'V1':
            raise ErpValidationError(f'unknown partner {order.partner_ref}')
        existing = self.orders.get(order.idempotency_key)
        if existing is not None:
            return existing
        number = len(self.orders) + 1
        created = PurchaseOrder(
            ref=f'PO{number:04d}',
            name=f'P{number:05d}',
            partner_ref=order.partner_ref,
            state=PurchaseOrderState.DRAFT,
            lines=order.lines,
        )
        self.orders[order.idempotency_key] = created
        return created

    async def get_purchase_order(self, po_ref: str) -> PurchaseOrder:
        for order in self.orders.values():
            if order.ref == po_ref:
                return order
        raise ErpNotFound(f'purchase order {po_ref}')


def tool_by_name(tools: Sequence[BaseTool], name: str) -> BaseTool:
    for candidate in tools:
        if candidate.name == name:
            return candidate
    raise AssertionError(f'tool {name} not found')


async def call(erp: FakeErp, name: str, args: dict[str, Any]) -> str:
    result = await tool_by_name(build_procurement_tools(erp), name).ainvoke(args)
    assert isinstance(result, str)
    return result


DRAFT_ARGS: dict[str, Any] = {
    'partner_ref': 'V1',
    'lines': [{'product_ref': 'P1', 'product_qty': '10', 'price_unit': '2.5'}],
}


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
