from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _lines(name: str) -> list[str]:
    return [line.strip() for line in (ROOT / name).read_text().splitlines()]


def test_image_runs_as_a_non_root_user() -> None:
    users = [line for line in _lines('Dockerfile') if line.startswith('USER ')]
    assert users
    assert users[-1].split()[1] not in {'root', '0'}


def test_base_images_are_pinned_to_python_3_12() -> None:
    bases = [line for line in _lines('Dockerfile') if line.startswith('FROM ')]
    assert bases
    assert all(line.split()[1].startswith('python:3.12') for line in bases)


def test_strict_msgpack_is_enabled() -> None:
    assert 'ENV LANGGRAPH_STRICT_MSGPACK=true' in _lines('Dockerfile')


def test_start_command_uses_the_app_factory() -> None:
    text = (ROOT / 'Dockerfile').read_text()
    assert 'agentic_ops.api.app:create_default_app' in text
    assert '--factory' in text


def test_dockerignore_keeps_secrets_and_history_out() -> None:
    entries = set(_lines('.dockerignore'))
    assert {'.env', '.git', '.venv'} <= entries
