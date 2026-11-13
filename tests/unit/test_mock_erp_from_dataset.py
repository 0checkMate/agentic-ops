"""MockErp seeded from the generated demo dataset."""

import pytest

from agentic_ops.erp.errors import ErpNotFound
from agentic_ops.erp.mock.adapter import MockErp
from agentic_ops.erp.mock.demo_seed import build_demo_dataset

DATASET = build_demo_dataset()


async def test_catalogue_matches_dataset() -> None:
    erp = MockErp.from_dataset(DATASET)
    products = await erp.list_products(limit=1000)
    assert list(products) == sorted(DATASET.products, key=lambda p: p.ref)


async def test_stock_matches_dataset() -> None:
    erp = MockErp.from_dataset(DATASET)
    assert tuple(await erp.get_stock_levels()) == tuple(DATASET.stock)


async def test_stock_can_be_filtered_by_product() -> None:
    erp = MockErp.from_dataset(DATASET)
    product_ref = next(iter(DATASET.stock)).product_ref
    levels = await erp.get_stock_levels(product_ref)
    assert levels
    assert all(level.product_ref == product_ref for level in levels)


async def test_every_partner_is_retrievable() -> None:
    erp = MockErp.from_dataset(DATASET)
    for partner in DATASET.partners:
        assert await erp.get_vendor(partner.ref) == partner


async def test_unknown_product_is_still_rejected() -> None:
    erp = MockErp.from_dataset(DATASET)
    with pytest.raises(ErpNotFound):
        await erp.get_stock_levels('NOPE-000')
