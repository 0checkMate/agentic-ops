"""Traceability seed: structure, time order, conservation and recall ground truth."""

import hashlib
from collections import defaultdict
from decimal import Decimal

import pytest

from agentic_ops.erp.mock.demo_seed import ANCHOR, DemoDataset, build_demo_dataset

# Hash of the pre-traceability data for seed 42. Change only on a deliberate dataset break.
EARLY_FINGERPRINT = '45f60cfe5291ef4fbd3d4a7f484c89dc8ceb3658cb268bdb68ed943599f05c1d'


@pytest.fixture(scope='module')
def dataset() -> DemoDataset:
    return build_demo_dataset()


def _early_fingerprint(data: DemoDataset) -> str:
    truth = data.truth
    payload = repr(
        (
            data.products,
            data.partners,
            data.stock,
            data.purchase_orders,
            data.receipts,
            data.invoices,
            sorted(data.reorder_points.items()),
            sorted(truth.below_reorder),
            sorted(truth.invoice_discrepancies.items()),
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _closure(data: DemoDataset, lot_ref: str) -> tuple[frozenset[str], frozenset[str]]:
    """Forward closure of a supplier lot, computed from the records alone."""
    batch_refs = {
        batch.ref
        for batch in data.batches
        if any(use.supplier_lot_ref == lot_ref for use in batch.lines)
    }
    finished_refs = {
        finished.ref for finished in data.finished_lots if finished.batch_ref in batch_refs
    }
    customer_refs = {
        shipment.customer_ref
        for shipment in data.shipments
        if any(item.finished_lot_ref in finished_refs for item in shipment.lines)
    }
    return frozenset(finished_refs), frozenset(customer_refs)


def test_early_data_is_unchanged(dataset: DemoDataset) -> None:
    assert _early_fingerprint(dataset) == EARLY_FINGERPRINT


def test_same_seed_gives_identical_dataset() -> None:
    assert build_demo_dataset(7) == build_demo_dataset(7)


def test_references_resolve_and_refs_are_unique(dataset: DemoDataset) -> None:
    receipts = {receipt.ref for receipt in dataset.receipts}
    lots = {supplier_lot.ref for supplier_lot in dataset.supplier_lots}
    batches = {batch.ref for batch in dataset.batches}
    finished = {finished_lot.ref for finished_lot in dataset.finished_lots}
    customers = {customer.ref for customer in dataset.customers}
    assert len(lots) == len(dataset.supplier_lots)
    assert len(batches) == len(dataset.batches)
    assert len(finished) == len(dataset.finished_lots)
    assert len(customers) == len(dataset.customers)
    assert all(supplier_lot.receipt_ref in receipts for supplier_lot in dataset.supplier_lots)
    assert all(use.supplier_lot_ref in lots for b in dataset.batches for use in b.lines)
    assert all(finished_lot.batch_ref in batches for finished_lot in dataset.finished_lots)
    assert all(shipment.customer_ref in customers for shipment in dataset.shipments)
    assert all(item.finished_lot_ref in finished for s in dataset.shipments for item in s.lines)


def test_time_order_holds(dataset: DemoDataset) -> None:
    receipts = {receipt.ref: receipt for receipt in dataset.receipts}
    lots = {supplier_lot.ref: supplier_lot for supplier_lot in dataset.supplier_lots}
    batches = {batch.ref: batch for batch in dataset.batches}
    finished = {finished_lot.ref: finished_lot for finished_lot in dataset.finished_lots}
    for supplier_lot in dataset.supplier_lots:
        assert supplier_lot.received_at == receipts[supplier_lot.receipt_ref].received_at
    for batch in dataset.batches:
        assert batch.started_at <= ANCHOR
        for use in batch.lines:
            assert batch.started_at >= lots[use.supplier_lot_ref].received_at
    for finished_lot in dataset.finished_lots:
        assert finished_lot.produced_at <= ANCHOR
        assert finished_lot.produced_at >= batches[finished_lot.batch_ref].started_at
    for shipment in dataset.shipments:
        assert shipment.shipped_at <= ANCHOR
        for item in shipment.lines:
            assert shipment.shipped_at >= finished[item.finished_lot_ref].produced_at


def test_quantities_are_conserved(dataset: DemoDataset) -> None:
    consumed: defaultdict[str, Decimal] = defaultdict(Decimal)
    for batch in dataset.batches:
        for use in batch.lines:
            consumed[use.supplier_lot_ref] += use.quantity
    for supplier_lot in dataset.supplier_lots:
        assert consumed[supplier_lot.ref] <= supplier_lot.quantity
    shipped: defaultdict[str, Decimal] = defaultdict(Decimal)
    for shipment in dataset.shipments:
        for item in shipment.lines:
            shipped[item.finished_lot_ref] += item.quantity
    for finished_lot in dataset.finished_lots:
        assert shipped[finished_lot.ref] <= finished_lot.quantity


def test_supplier_lots_mirror_inbound_receipt_lines(dataset: DemoDataset) -> None:
    """One lot per ingredient or packaging receipt line; chemicals are never lotted."""
    codes = {product.ref: product.default_code or '' for product in dataset.products}
    expected = {
        (receipt.ref, item.product_ref): item.quantity
        for receipt in dataset.receipts
        for item in receipt.lines
        if codes[item.product_ref].startswith(('ING-', 'PKG-'))
    }
    actual = {
        (supplier_lot.receipt_ref, supplier_lot.product_ref): supplier_lot.quantity
        for supplier_lot in dataset.supplier_lots
    }
    assert len(dataset.supplier_lots) == len(actual)
    assert actual == expected


def test_truth_equals_independent_forward_closure(dataset: DemoDataset) -> None:
    scope = dataset.truth.recall_scope
    assert set(scope) == {supplier_lot.ref for supplier_lot in dataset.supplier_lots}
    for lot_ref, expected in scope.items():
        finished_refs, customer_refs = _closure(dataset, lot_ref)
        assert expected.finished_lots == finished_refs
        assert expected.customers == customer_refs


def test_graph_contains_the_planted_shapes(dataset: DemoDataset) -> None:
    batch_of = {finished_lot.ref: finished_lot.batch_ref for finished_lot in dataset.finished_lots}
    customers_per_finished: defaultdict[str, set[str]] = defaultdict(set)
    batches_per_customer: defaultdict[str, set[str]] = defaultdict(set)
    for shipment in dataset.shipments:
        for item in shipment.lines:
            customers_per_finished[item.finished_lot_ref].add(shipment.customer_ref)
            batches_per_customer[shipment.customer_ref].add(batch_of[item.finished_lot_ref])
    scope = dataset.truth.recall_scope.values()
    assert any(len(batch.lines) >= 2 for batch in dataset.batches)  # blending
    assert any(len(customers) >= 2 for customers in customers_per_finished.values())  # split
    assert any(len(batch_refs) >= 2 for batch_refs in batches_per_customer.values())  # fan-in
    assert any(not entry.finished_lots for entry in scope)  # dead end: never consumed
    assert any(entry.finished_lots and not entry.customers for entry in scope)  # unshipped
    assert any(entry.customers for entry in scope)  # a real recall
