from collections.abc import Iterable, Sequence

from agentic_ops.erp.errors import ErpNotFound, ErpValidationError
from agentic_ops.erp.models import (
    NewPurchaseOrder,
    Partner,
    Product,
    PurchaseOrder,
    PurchaseOrderState,
    StockLevel,
)


class MockErp:
    """In-memory ERP seeded with synthetic data.

    Mirrors Odoo's entities and field names, not its full behaviour.
    """

    def __init__(
        self,
        *,
        products: Iterable[Product],
        stock: Iterable[StockLevel],
        partners: Iterable[Partner],
    ) -> None:
        self._products = {p.ref: p for p in products}
        self._stock = tuple(stock)
        self._partners = {p.ref: p for p in partners}
        self._orders: dict[str, PurchaseOrder] = {}
        self._order_ref_by_key: dict[str, str] = {}

    async def list_products(self, *, limit: int = 100, offset: int = 0) -> Sequence[Product]:
        ordered = sorted(self._products.values(), key=lambda p: p.ref)
        return ordered[offset : offset + limit]

    async def get_stock_levels(self, product_ref: str | None = None) -> Sequence[StockLevel]:
        if product_ref is None:
            return self._stock
        if product_ref not in self._products:
            raise ErpNotFound(f'Product {product_ref} not found')
        return tuple(s for s in self._stock if s.product_ref == product_ref)

    async def get_vendor(self, partner_ref: str) -> Partner:
        partner = self._partners.get(partner_ref)
        if partner is None:
            raise ErpNotFound(f'Partner {partner_ref} not found')
        return partner

    async def create_draft_po(self, order: NewPurchaseOrder) -> PurchaseOrder:
        existing_ref = self._order_ref_by_key.get(order.idempotency_key)
        if existing_ref is not None:
            return self._orders[existing_ref]
        if order.partner_ref not in self._partners:
            raise ErpValidationError(f'Unknown partner {order.partner_ref}')
        for line in order.lines:
            if line.product_ref not in self._products:
                raise ErpValidationError(f'Unknown product {line.product_ref}')
        ref = f'PO-{len(self._orders) + 1:03d}'
        created = PurchaseOrder(
            ref=ref,
            name=ref,
            partner_ref=order.partner_ref,
            state=PurchaseOrderState.DRAFT,
            lines=order.lines,
        )
        self._orders[ref] = created
        self._order_ref_by_key[order.idempotency_key] = ref
        return created

    async def get_purchase_order(self, po_ref: str) -> PurchaseOrder:
        order = self._orders.get(po_ref)
        if order is None:
            raise ErpNotFound(f'Purchase order {po_ref} not found')
        return order
