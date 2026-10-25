import pytest

from agentic_ops.db.checkpoint import to_psycopg_url


def test_strips_the_asyncpg_driver() -> None:
    assert to_psycopg_url('postgresql+asyncpg://u:p@host/db') == 'postgresql://u:p@host/db'


def test_plain_url_is_unchanged() -> None:
    url = 'postgresql://u:p@host:5432/db'
    assert to_psycopg_url(url) == url


def test_ssl_parameter_becomes_sslmode() -> None:
    assert (
        to_psycopg_url('postgresql+asyncpg://u:p@host/db?ssl=require')
        == 'postgresql://u:p@host/db?sslmode=require'
    )


def test_sslmode_is_kept() -> None:
    url = 'postgresql://u:p@host/db?sslmode=require'
    assert to_psycopg_url(url) == url


def test_other_schemes_are_rejected() -> None:
    with pytest.raises(ValueError):
        to_psycopg_url('mysql://u:p@host/db')
