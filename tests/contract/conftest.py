import pytest

from agentic_ops.erp.port import ErpPort
from tests.contract.seed import SEED_PARTNERS, SEED_PRODUCTS, SEED_STOCK


def _make_mock() -> ErpPort:
    from agentic_ops.erp.mock import MockErp

    return MockErp(products=SEED_PRODUCTS, stock=SEED_STOCK, partners=SEED_PARTNERS)


@pytest.fixture(params=['mock'])
def erp(request: pytest.FixtureRequest) -> ErpPort:
    factories = {'mock': _make_mock}
    return factories[request.param]()
