from pydantic import SecretStr

from agentic_ops.config import Settings


def test_sqlalchemy_url_selects_psycopg_driver() -> None:
    settings = Settings(database_url=SecretStr('postgresql://u:p@h/db'))
    assert settings.sqlalchemy_url == 'postgresql+psycopg://u:p@h/db'


def test_database_url_is_not_leaked_in_repr() -> None:
    settings = Settings(database_url=SecretStr('postgresql://u:topsecret@h/db'))
    assert 'topsecret' not in repr(settings)
