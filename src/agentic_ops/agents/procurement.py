"""Procurement specialist: thin ErpPort tools and the agent-to-specialist adapter."""

import hashlib
import json
from collections.abc import Awaitable, Sequence
from decimal import Decimal
from typing import Annotated, Any

from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from agentic_ops.agents.supervisor import AgentState, Specialist
from agentic_ops.erp.errors import ErpNotFound, ErpValidationError
from agentic_ops.erp.models import NewPurchaseOrder, PurchaseOrderLine
from agentic_ops.erp.port import ErpPort

MAX_PAGE_SIZE = 50


class DraftLine(BaseModel):
    """One purchase order line, as supplied by the model."""

    product_ref: str
    product_qty: Decimal = Field(gt=0)
    price_unit: Decimal = Field(ge=0)


def _json(models: BaseModel | Sequence[BaseModel]) -> str:
    if isinstance(models, BaseModel):
        return models.model_dump_json()
    return '[' + ','.join(m.model_dump_json() for m in models) + ']'


def _plain(value: Decimal) -> str:
    return format(value.normalize(), 'f')


def _idempotency_key(partner_ref: str, lines: Sequence[DraftLine]) -> str:
    canonical = json.dumps(
        [
            partner_ref,
            [[ln.product_ref, _plain(ln.product_qty), _plain(ln.price_unit)] for ln in lines],
        ],
        separators=(',', ':'),
    )
    return 'agent-' + hashlib.sha256(canonical.encode()).hexdigest()[:32]


async def _guard(call: Awaitable[str]) -> str:
    try:
        return await call
    except (ErpNotFound, ErpValidationError) as exc:
        return f'ERROR: {exc}'


def build_procurement_tools(erp: ErpPort) -> list[BaseTool]:
    """Build the procurement toolset over any ErpPort adapter."""

    @tool
    async def list_products(limit: int = MAX_PAGE_SIZE, offset: int = 0) -> str:
        """List products (reference, name, unit of measure), one page at a time."""
        page = min(max(limit, 1), MAX_PAGE_SIZE)
        products = await erp.list_products(limit=page, offset=max(offset, 0))
        return _json(products)

    @tool
    async def get_stock_levels(product_ref: str | None = None) -> str:
        """Get stock levels for one product reference, or for all products if omitted."""

        async def run() -> str:
            return _json(await erp.get_stock_levels(product_ref))

        return await _guard(run())

    @tool
    async def get_vendor(partner_ref: str) -> str:
        """Get a supplier by its partner reference."""

        async def run() -> str:
            return _json(await erp.get_vendor(partner_ref))

        return await _guard(run())

    @tool
    async def create_draft_po(
        partner_ref: str, lines: Annotated[list[DraftLine], Field(min_length=1)]
    ) -> str:
        """Create a DRAFT purchase order. A human must confirm it in the ERP."""

        async def run() -> str:
            order = NewPurchaseOrder(
                partner_ref=partner_ref,
                lines=tuple(
                    PurchaseOrderLine(
                        product_ref=ln.product_ref,
                        product_qty=ln.product_qty,
                        price_unit=ln.price_unit,
                    )
                    for ln in lines
                ),
                idempotency_key=_idempotency_key(partner_ref, lines),
            )
            return _json(await erp.create_draft_po(order))

        return await _guard(run())

    @tool
    async def get_purchase_order(po_ref: str) -> str:
        """Get a purchase order by its reference."""

        async def run() -> str:
            return _json(await erp.get_purchase_order(po_ref))

        return await _guard(run())

    return [list_products, get_stock_levels, get_vendor, create_draft_po, get_purchase_order]


def agent_as_specialist(agent: Runnable[Any, Any]) -> Specialist:
    """Wrap a compiled agent as a supervisor node that returns only what it added."""

    async def node(state: AgentState) -> dict[str, Any]:
        history = list(state['messages'])
        result = await agent.ainvoke({'messages': history})
        return {'messages': list(result['messages'])[len(history) :]}

    return node
