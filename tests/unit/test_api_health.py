from collections.abc import Awaitable, Callable

import httpx
from httpx import ASGITransport

from agentic_ops.api.app import create_app


def _client(check: Callable[[], Awaitable[bool]]) -> httpx.AsyncClient:
    app = create_app(db_check=check)
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url='http://test')


async def _ok() -> bool:
    return True


async def _down() -> bool:
    return False


async def _boom() -> bool:
    raise RuntimeError('secret-host.internal refused the connection')


async def test_health_reports_ok() -> None:
    async with _client(_ok) as client:
        response = await client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


async def test_health_never_touches_the_database() -> None:
    async with _client(_boom) as client:
        response = await client.get('/health')
    assert response.status_code == 200


async def test_ready_is_200_when_the_database_answers() -> None:
    async with _client(_ok) as client:
        response = await client.get('/ready')
    assert response.status_code == 200
    assert response.json() == {'status': 'ready'}


async def test_ready_is_503_when_the_check_fails() -> None:
    async with _client(_down) as client:
        response = await client.get('/ready')
    assert response.status_code == 503
    assert response.json() == {'status': 'unavailable'}


async def test_ready_hides_the_error_detail() -> None:
    async with _client(_boom) as client:
        response = await client.get('/ready')
    assert response.status_code == 503
    assert 'secret-host' not in response.text
