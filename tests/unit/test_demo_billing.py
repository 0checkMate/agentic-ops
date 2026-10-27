from collections import defaultdict
from decimal import Decimal

from agentic_ops.erp.mock.demo_seed import (
    ANCHOR,
    PRICE_TOLERANCE,
    DemoDataset,
    DiscrepancyKind,
    build_demo_dataset,
)
from agentic_ops.erp.models import PurchaseOrder, PurchaseOrderState, SupplierInvoice

_BILLABLE = {PurchaseOrderState.PURCHASE, PurchaseOrderState.DONE}


def _classify(data: DemoDataset, invoice: SupplierInvoice) -> DiscrepancyKind | None:
    """Reference three-way match. The generator's planted truth must agree with it."""
    order = next(item for item in data.purchase_orders if item.ref == invoice.purchase_order_ref)
    if any(
        other.partner_ref == invoice.partner_ref
        and other.number == invoice.number
        and other.ref < invoice.ref
        for other in data.invoices
    ):
        return DiscrepancyKind.DUPLICATE
    receipts = [item for item in data.receipts if item.purchase_order_ref == order.ref]
    if not receipts:
        return DiscrepancyKind.NO_RECEIPT
    received: defaultdict[str, Decimal] = defaultdict(Decimal)
    for receipt in receipts:
        for received_line in receipt.lines:
            received[received_line.product_ref] += received_line.quantity
    prices = {line.product_ref: line.price_unit for line in order.lines}
    if any(line.product_ref not in prices for line in invoice.lines):
        return DiscrepancyKind.UNORDERED_ITEM
    if any(line.quantity > received[line.product_ref] for line in invoice.lines):
        return DiscrepancyKind.QUANTITY_MISMATCH
    for line in invoice.lines:
        expected = prices[line.product_ref]
        if abs(line.price_unit - expected) > expected * PRICE_TOLERANCE:
            return DiscrepancyKind.PRICE_MISMATCH
    return None


def _clean(data: DemoDataset) -> list[tuple[SupplierInvoice, PurchaseOrder]]:
    orders = {order.ref: order for order in data.purchase_orders}
    return [
        (invoice, orders[invoice.purchase_order_ref])
        for invoice in data.invoices
        if invoice.ref not in data.truth.invoice_discrepancies
    ]


def test_reference_matcher_agrees_with_planted_truth() -> None:
    data = build_demo_dataset()
    for invoice in data.invoices:
        assert _classify(data, invoice) == data.truth.invoice_discrepancies.get(invoice.ref)


def test_truth_only_names_real_invoices() -> None:
    data = build_demo_dataset()
    assert set(data.truth.invoice_discrepancies) <= {invoice.ref for invoice in data.invoices}


def test_every_discrepancy_kind_is_planted() -> None:
    data = build_demo_dataset()
    assert set(data.truth.invoice_discrepancies.values()) == set(DiscrepancyKind)


def test_most_invoices_are_clean() -> None:
    data = build_demo_dataset()
    assert len(data.truth.invoice_discrepancies) * 2 < len(data.invoices)


def test_clean_invoices_include_price_noise_within_tolerance() -> None:
    noisy = [
        line
        for invoice, order in _clean(build_demo_dataset())
        for line in invoice.lines
        for ordered in order.lines
        if ordered.product_ref == line.product_ref and ordered.price_unit != line.price_unit
    ]
    assert noisy


def test_clean_invoices_include_short_deliveries() -> None:
    short = [
        line
        for invoice, order in _clean(build_demo_dataset())
        for line in invoice.lines
        for ordered in order.lines
        if ordered.product_ref == line.product_ref and line.quantity < ordered.product_qty
    ]
    assert short


def test_documents_hang_off_billable_orders() -> None:
    data = build_demo_dataset()
    orders = {order.ref: order for order in data.purchase_orders}
    for receipt in data.receipts:
        assert orders[receipt.purchase_order_ref].state in _BILLABLE
    for invoice in data.invoices:
        order = orders[invoice.purchase_order_ref]
        assert order.state in _BILLABLE
        assert invoice.partner_ref == order.partner_ref


def test_documents_follow_their_order_and_precede_the_anchor() -> None:
    data = build_demo_dataset()
    orders = {order.ref: order for order in data.purchase_orders}
    for receipt in data.receipts:
        placed = orders[receipt.purchase_order_ref].date_order
        assert placed is not None
        assert placed < receipt.received_at <= ANCHOR
    for invoice in data.invoices:
        placed = orders[invoice.purchase_order_ref].date_order
        assert placed is not None
        assert placed < invoice.invoice_date <= ANCHOR


def test_document_references_are_unique() -> None:
    data = build_demo_dataset()
    for refs in (
        [receipt.ref for receipt in data.receipts],
        [invoice.ref for invoice in data.invoices],
    ):
        assert len(refs) == len(set(refs))


def test_billing_scale() -> None:
    data = build_demo_dataset()
    assert len(data.receipts) >= 60
    assert 60 <= len(data.invoices) <= 200


def test_receipts_never_exceed_the_ordered_quantity() -> None:
    data = build_demo_dataset()
    received: defaultdict[tuple[str, str], Decimal] = defaultdict(Decimal)
    for receipt in data.receipts:
        for line in receipt.lines:
            received[(receipt.purchase_order_ref, line.product_ref)] += line.quantity
    ordered = {
        (order.ref, line.product_ref): line.product_qty
        for order in data.purchase_orders
        for line in order.lines
    }
    for key, quantity in received.items():
        assert quantity <= ordered[key]
