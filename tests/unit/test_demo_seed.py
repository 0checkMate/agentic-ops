from decimal import Decimal

from agentic_ops.erp.mock.demo_seed import ANCHOR, build_demo_dataset
from agentic_ops.erp.models import PurchaseOrderState


def test_dataset_is_deterministic() -> None:
    assert build_demo_dataset(seed=1) == build_demo_dataset(seed=1)


def test_different_seeds_differ() -> None:
    assert build_demo_dataset(seed=1) != build_demo_dataset(seed=2)


def test_dataset_scale() -> None:
    data = build_demo_dataset()
    suppliers = [partner for partner in data.partners if partner.is_supplier]
    assert 25 <= len(data.products) <= 40
    assert 10 <= len(suppliers) <= 14
    assert 150 <= len(data.purchase_orders) <= 300


def test_references_are_unique() -> None:
    data = build_demo_dataset()
    groups = (
        [product.ref for product in data.products],
        [partner.ref for partner in data.partners],
        [order.ref for order in data.purchase_orders],
        [order.name for order in data.purchase_orders],
    )
    for refs in groups:
        assert len(refs) == len(set(refs))


def test_orders_reference_known_suppliers_and_products() -> None:
    data = build_demo_dataset()
    suppliers = {partner.ref for partner in data.partners if partner.is_supplier}
    products = {product.ref for product in data.products}
    for order in data.purchase_orders:
        assert order.partner_ref in suppliers
        assert all(line.product_ref in products for line in order.lines)


def test_every_storable_product_has_stock_and_a_reorder_point() -> None:
    data = build_demo_dataset()
    storable = {product.ref for product in data.products if product.is_storable}
    assert {level.product_ref for level in data.stock} == storable
    assert set(data.reorder_points) == storable


def test_every_order_state_is_represented() -> None:
    data = build_demo_dataset()
    assert {order.state for order in data.purchase_orders} == set(PurchaseOrderState)


def test_orders_are_dated_on_or_before_the_anchor() -> None:
    data = build_demo_dataset()
    for order in data.purchase_orders:
        assert order.date_order is not None
        assert order.date_order <= ANCHOR


def test_truth_matches_stock_and_reorder_points() -> None:
    data = build_demo_dataset()
    free: dict[str, Decimal] = {}
    for level in data.stock:
        free[level.product_ref] = (
            free.get(level.product_ref, Decimal('0')) + level.quantity - level.reserved_quantity
        )
    expected = frozenset(
        ref for ref, point in data.reorder_points.items() if free.get(ref, Decimal('0')) < point
    )
    assert data.truth.below_reorder == expected


def test_some_but_not_all_products_are_below_reorder() -> None:
    data = build_demo_dataset()
    assert 3 <= len(data.truth.below_reorder) < len(data.reorder_points)
