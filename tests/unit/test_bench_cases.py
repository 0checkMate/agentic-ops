from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from agentic_ops.bench.cases import CaseSuite, load_suite

SHIPPED = Path(__file__).resolve().parents[2] / 'benchmarks' / 'procurement_v1.json'


def tool_def() -> dict[str, Any]:
    return {
        'name': 'get_stock_level',
        'description': 'Look up current stock for a SKU.',
        'parameters': {
            'type': 'object',
            'properties': {'sku': {'type': 'string'}, 'warehouse': {'type': 'string'}},
            'required': ['sku'],
        },
    }


def call_case(case_id: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        'id': case_id,
        'prompt': 'How much stock do we have?',
        'expected': {'name': name, 'args': args},
    }


def abstain_case(case_id: str) -> dict[str, Any]:
    return {'id': case_id, 'prompt': 'What is HACCP?', 'expected': None}


def suite_data(*cases: dict[str, Any]) -> dict[str, Any]:
    return {'version': 'test-1', 'tools': [tool_def()], 'cases': list(cases)}


def test_valid_suite_loads() -> None:
    suite = CaseSuite.model_validate(suite_data(call_case('c1', 'get_stock_level', {'sku': 'A'})))
    assert suite.cases[0].expected is not None
    assert suite.cases[0].expected.name == 'get_stock_level'


def test_abstain_case_is_allowed() -> None:
    suite = CaseSuite.model_validate(suite_data(abstain_case('c1')))
    assert suite.cases[0].expected is None


def test_duplicate_case_ids_are_rejected() -> None:
    data = suite_data(
        call_case('c1', 'get_stock_level', {'sku': 'A'}),
        call_case('c1', 'get_stock_level', {'sku': 'B'}),
    )
    with pytest.raises(ValidationError, match='Duplicate case id'):
        CaseSuite.model_validate(data)


def test_unknown_expected_tool_is_rejected() -> None:
    data = suite_data(call_case('c1', 'no_such_tool', {'sku': 'A'}))
    with pytest.raises(ValidationError, match='Unknown tool'):
        CaseSuite.model_validate(data)


def test_argument_outside_schema_is_rejected() -> None:
    data = suite_data(call_case('c1', 'get_stock_level', {'sku': 'A', 'colour': 'red'}))
    with pytest.raises(ValidationError, match='not in schema'):
        CaseSuite.model_validate(data)


def test_missing_required_argument_is_rejected() -> None:
    data = suite_data(call_case('c1', 'get_stock_level', {'warehouse': 'W1'}))
    with pytest.raises(ValidationError, match='Missing required'):
        CaseSuite.model_validate(data)


def test_tool_converts_to_openai_function_format() -> None:
    suite = CaseSuite.model_validate(suite_data(abstain_case('c1')))
    assert suite.tools[0].to_openai() == {
        'type': 'function',
        'function': {
            'name': 'get_stock_level',
            'description': 'Look up current stock for a SKU.',
            'parameters': tool_def()['parameters'],
        },
    }


def test_load_suite_reads_a_file(tmp_path: Path) -> None:
    import json

    path = tmp_path / 'suite.json'
    path.write_text(json.dumps(suite_data(abstain_case('c1'))), encoding='utf-8')
    assert load_suite(path).version == 'test-1'


def test_shipped_suite_is_valid_and_exercises_every_tool() -> None:
    suite = load_suite(SHIPPED)
    assert len(suite.cases) >= 20
    assert sum(item.expected is None for item in suite.cases) >= 4
    used = {item.expected.name for item in suite.cases if item.expected is not None}
    assert used == {tool.name for tool in suite.tools}
