import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncEngine

from agentic_ops.config import get_settings
from agentic_ops.db.session import make_engine

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def test_db_url(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    url = os.environ.get('AGENTIC_TEST_DATABASE_URL')
    if not url:
        if os.environ.get('CI'):
            pytest.fail('AGENTIC_TEST_DATABASE_URL must be set in CI')
        pytest.skip('AGENTIC_TEST_DATABASE_URL is not set')
    if 'neon.tech' in url:
        pytest.fail('Refusing to run destructive migration tests against a Neon database')
    monkeypatch.setenv('AGENTIC_DATABASE_URL', url)
    get_settings.cache_clear()
    yield url
    get_settings.cache_clear()


@pytest.fixture
def alembic_config(test_db_url: str) -> Config:
    config = Config(str(ROOT / 'alembic.ini'))
    config.set_main_option('script_location', str(ROOT / 'migrations'))
    return config


@pytest.fixture
def migrated_db(alembic_config: Config) -> Config:
    """Bring the test database to head; a no-op if it is already there."""
    command.upgrade(alembic_config, 'head')
    return alembic_config


@pytest.fixture
async def async_engine(migrated_db: Config) -> AsyncIterator[AsyncEngine]:
    """Per-test engine with NullPool: each test has its own event loop."""
    engine = make_engine(get_settings().sqlalchemy_url, null_pool=True)
    try:
        yield engine
    finally:
        await engine.dispose()
