from decimal import Decimal

import pytest

from agentic_ops.erp.errors import ErpNotFound, ErpValidationError
from agentic_ops.erp.models import NewPurchaseOrder, PurchaseOrderLine, PurchaseOrderState
from agentic_ops.erp.port import ErpPort
from tests.contract.seed import SEED_PARTNERS, SEED_PRODUCTS, SEED_STOCK

pytestmark = pytest.mark.contract


def _new_order(
    key: str = 'key-1', partner_ref: str = 'V-001', product_ref: str = 'P-001'
) -> NewPurchaseOrder:
    line = PurchaseOrderLine(
        product_ref=product_ref, product_qty=Decimal('50'), price_unit=Decimal('2.5')
    )
    return NewPurchaseOrder(partner_ref=partner_ref, lines=(line,), idempotency_key=key)


async def test_list_products_is_ordered_by_ref(erp: ErpPort) -> None:
    products = await erp.list_products()
    assert [p.ref for p in products] == sorted(p.ref for p in SEED_PRODUCTS)


async def test_list_products_pagination(erp: ErpPort) -> None:
    expected = sorted(p.ref for p in SEED_PRODUCTS)
    page = await erp.list_products(limit=1, offset=1)
    assert [p.ref for p in page] == [expected[1]]


async def test_stock_levels_for_all_products(erp: ErpPort) -> None:
    levels = await erp.get_stock_levels()
    assert set(levels) == set(SEED_STOCK)


async def test_stock_levels_for_one_product(erp: ErpPort) -> None:
    levels = await erp.get_stock_levels('P-001')
    assert set(levels) == {s for s in SEED_STOCK if s.product_ref == 'P-001'}
    assert len(levels) == 2


async def test_stock_levels_for_product_without_stock_is_empty(erp: ErpPort) -> None:
    assert len(await erp.get_stock_levels('P-003')) == 0


async def test_stock_levels_unknown_product_raises(erp: ErpPort) -> None:
    with pytest.raises(ErpNotFound):
        await erp.get_stock_levels('P-999')


async def test_get_vendor(erp: ErpPort) -> None:
    vendor = await erp.get_vendor('V-001')
    assert vendor == SEED_PARTNERS[0]


async def test_get_vendor_unknown_raises(erp: ErpPort) -> None:
    with pytest.raises(ErpNotFound):
        await erp.get_vendor('V-999')


async def test_create_draft_po_round_trip(erp: ErpPort) -> None:
    order = _new_order()
    created = await erp.create_draft_po(order)
    assert created.state == PurchaseOrderState.DRAFT
    assert created.partner_ref == 'V-001'
    assert created.lines == order.lines
    assert await erp.get_purchase_order(created.ref) == created


async def test_create_draft_po_is_idempotent(erp: ErpPort) -> None:
    first = await erp.create_draft_po(_new_order(key='key-1'))
    repeat = await erp.create_draft_po(_new_order(key='key-1'))
    other = await erp.create_draft_po(_new_order(key='key-2'))
    assert repeat.ref == first.ref
    assert other.ref != first.ref


async def test_create_draft_po_unknown_partner_raises(erp: ErpPort) -> None:
    with pytest.raises(ErpValidationError):
        await erp.create_draft_po(_new_order(partner_ref='V-999'))


async def test_create_draft_po_unknown_product_raises(erp: ErpPort) -> None:
    with pytest.raises(ErpValidationError):
        await erp.create_draft_po(_new_order(product_ref='P-999'))


async def test_get_purchase_order_unknown_raises(erp: ErpPort) -> None:
    with pytest.raises(ErpNotFound):
        await erp.get_purchase_order('PO-999')


def test_adapter_exposes_no_confirmation_method(erp: ErpPort) -> None:
    public = [name for name in dir(erp) if not name.startswith('_')]
    assert not [n for n in public if 'confirm' in n or 'approve' in n]
