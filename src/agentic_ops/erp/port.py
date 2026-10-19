from collections.abc import Sequence
from typing import Protocol

from agentic_ops.erp.models import (
    NewPurchaseOrder,
    Partner,
    Product,
    PurchaseOrder,
    StockLevel,
)


class ErpPort(Protocol):
    """Contract every ERP adapter must satisfy.

    The port deliberately has no method to confirm a purchase order: agents can
    only create drafts, and a human confirms them in the ERP.
    """

    async def list_products(self, *, limit: int = 100, offset: int = 0) -> Sequence[Product]:
        """Return a page of products ordered by reference."""

    async def get_stock_levels(self, product_ref: str | None = None) -> Sequence[StockLevel]:
        """Return stock levels for one product, or for all products if None.

        Raises ErpNotFound if a given product_ref does not exist.
        """

    async def get_vendor(self, partner_ref: str) -> Partner:
        """Return a supplier by reference.

        Raises ErpNotFound if it does not exist.
        """

    async def create_draft_po(self, order: NewPurchaseOrder) -> PurchaseOrder:
        """Create a purchase order in draft state.

        Idempotent on order.idempotency_key: a repeated call with the same key
        returns the original order and creates no duplicate.
        Raises ErpValidationError if a partner or product reference is unknown.
        """

    async def get_purchase_order(self, po_ref: str) -> PurchaseOrder:
        """Return a purchase order by reference.

        Raises ErpNotFound if it does not exist.
        """
