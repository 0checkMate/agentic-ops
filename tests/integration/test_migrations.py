import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from agentic_ops.config import get_settings

pytestmark = pytest.mark.integration

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


def _alembic_config() -> Config:
    config = Config(str(ROOT / 'alembic.ini'))
    config.set_main_option('script_location', str(ROOT / 'migrations'))
    return config


def test_migrations_round_trip(test_db_url: str) -> None:
    config = _alembic_config()
    command.upgrade(config, 'head')
    command.downgrade(config, 'base')
    command.upgrade(config, 'head')

    engine = create_engine(get_settings().sqlalchemy_url)
    try:
        tables = inspect(engine).get_table_names()
    finally:
        engine.dispose()
    assert 'audit_log' in tables
