"""The production MockErp is built from the deterministic demo dataset."""

from agentic_ops.erp.mock.demo import demo_erp
from agentic_ops.erp.mock.demo_seed import build_demo_dataset
from agentic_ops.erp.port import ErpPort


async def test_catalogue_matches_the_demo_dataset() -> None:
    erp: ErpPort = demo_erp()
    products = await erp.list_products(limit=100)
    assert [p.ref for p in products] == [p.ref for p in build_demo_dataset().products]


async def test_stock_matches_the_demo_dataset() -> None:
    erp: ErpPort = demo_erp()
    stock = await erp.get_stock_levels()
    assert len(stock) == len(build_demo_dataset().stock)


async def test_suppliers_are_available() -> None:
    erp: ErpPort = demo_erp()
    vendor = await erp.get_vendor('V-001')
    assert vendor.name == 'Rift Valley Fresh Produce Ltd'


def test_each_call_returns_an_independent_instance() -> None:
    assert demo_erp() is not demo_erp()
