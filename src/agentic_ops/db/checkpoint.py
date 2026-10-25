from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

_ACCEPTED_SCHEMES = frozenset({'postgresql', 'postgres'})


def to_psycopg_url(url: str) -> str:
    """Convert a SQLAlchemy-style URL to the plain libpq form that psycopg expects.

    Drops the '+driver' suffix (for example '+asyncpg') and renames asyncpg's 'ssl'
    query parameter to libpq's 'sslmode'. The URL is never echoed in errors because it
    carries credentials.
    """
    parts = urlsplit(url)
    scheme = parts.scheme.split('+', 1)[0]
    if scheme not in _ACCEPTED_SCHEMES:
        raise ValueError(f'Unsupported database scheme {scheme!r}')
    query = urlencode(
        [
            ('sslmode' if key == 'ssl' else key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
        ]
    )
    return urlunsplit((scheme, parts.netloc, parts.path, query, parts.fragment))


@asynccontextmanager
async def open_checkpointer(database_url: str) -> AsyncIterator[AsyncPostgresSaver]:
    """Open the LangGraph Postgres checkpointer on its own connection, tables ensured.

    setup() is idempotent: the library creates and migrates its own tables, so they are
    deliberately not part of our Alembic history.
    """
    async with AsyncPostgresSaver.from_conn_string(to_psycopg_url(database_url)) as saver:
        await saver.setup()
        yield saver
