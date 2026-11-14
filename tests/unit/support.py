"""Shared test doubles for the agent tests: scripted model, fake ERP, helpers."""

from collections.abc import Sequence
from decimal import Decimal
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable

from agentic_ops.agents.supervisor import ROUTE_TOOL_NAME, AgentState, Specialist
from agentic_ops.erp.errors import ErpNotFound, ErpValidationError
from agentic_ops.erp.models import (
    NewPurchaseOrder,
    Partner,
    Product,
    PurchaseOrder,
    PurchaseOrderState,
    StockLevel,
)

DRAFT_ARGS: dict[str, Any] = {
    'partner_ref': 'V1',
    'lines': [{'product_ref': 'P1', 'product_qty': '10', 'price_unit': '2.5'}],
}


class ScriptedChatModel(BaseChatModel):
    """Replays scripted AIMessages in order and records what it was shown."""

    script: list[AIMessage]
    seen: list[list[BaseMessage]] = []
    cursor: int = 0

    @property
    def _llm_type(self) -> str:
        return 'scripted'

    def bind_tools(
        self, tools: Sequence[Any], **kwargs: Any
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen.append(list(messages))
        reply = self.script[self.cursor]
        self.cursor += 1
        return ChatResult(generations=[ChatGeneration(message=reply)])


def tool_call_message(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(
        content='',
        tool_calls=[{'name': name, 'args': args, 'id': call_id, 'type': 'tool_call'}],
    )


def route_call(agent: str) -> AIMessage:
    return tool_call_message(ROUTE_TOOL_NAME, {'agent': agent}, 'route_1')


def unpaired_tool_call_ids(messages: Sequence[BaseMessage]) -> set[str]:
    """Tool-call ids with no ToolMessage answer: providers reject such histories."""
    called = {
        call['id']
        for m in messages
        if isinstance(m, AIMessage)
        for call in m.tool_calls
        if call['id'] is not None
    }
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    return called - answered


def recording_specialist(name: str, calls: list[str]) -> Specialist:
    async def node(state: AgentState) -> dict[str, Any]:
        calls.append(name)
        return {'messages': [AIMessage(content=f'{name} done', name=name)]}

    return node


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
