import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from agentic_ops.config import get_settings

pytestmark = pytest.mark.integration


def test_migrations_round_trip(alembic_config: Config) -> None:
    command.upgrade(alembic_config, 'head')
    command.downgrade(alembic_config, 'base')
    command.upgrade(alembic_config, 'head')

    engine = create_engine(get_settings().sqlalchemy_url)
    try:
        tables = inspect(engine).get_table_names()
    finally:
        engine.dispose()
    assert {'audit_log', 'cache_entry'} <= set(tables)
