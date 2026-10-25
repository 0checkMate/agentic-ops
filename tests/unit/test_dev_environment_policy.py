import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _text(name: str) -> str:
    return (ROOT / name).read_text()


def test_compose_runs_the_same_postgres_image_as_ci() -> None:
    image = 'pgvector/pgvector:pg18'
    assert image in _text('compose.yaml')
    assert image in _text('.github/workflows/ci.yml')


def test_postgres_volume_uses_the_pg18_mount_point() -> None:
    text = _text('compose.yaml')
    mounts = [line.strip() for line in text.splitlines()]
    assert any(line.endswith(':/var/lib/postgresql') for line in mounts)
    assert '/var/lib/postgresql/data' not in text


def test_database_port_is_bound_to_loopback() -> None:
    assert '127.0.0.1:5432:5432' in _text('compose.yaml')


def test_test_database_is_created_on_first_start() -> None:
    script = _text('docker/initdb/01-create-test-db.sql')
    assert 'CREATE DATABASE agentic_test' in script


def test_devcontainer_targets_the_compose_workspace() -> None:
    config = json.loads(_text('.devcontainer/devcontainer.json'))
    assert isinstance(config, dict)
    assert config['dockerComposeFile'] == '../compose.yaml'
    assert config['service'] == 'workspace'
