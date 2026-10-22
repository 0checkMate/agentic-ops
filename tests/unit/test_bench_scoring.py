from typing import Any

import pytest

from agentic_ops.bench.scoring import (
    ExpectedCall,
    ObservedCall,
    Verdict,
    score_case,
    tool_call_accuracy,
)


def expected(name: str = 'get_stock_level', **args: Any) -> ExpectedCall:
    return ExpectedCall(name=name, args=dict(args))


def observed(name: str = 'get_stock_level', **args: Any) -> ObservedCall:
    return ObservedCall(name=name, args=dict(args))


def test_identical_call_is_correct() -> None:
    verdict = score_case(expected(sku='FLOUR-001'), [observed(sku='FLOUR-001')])
    assert verdict is Verdict.CORRECT


def test_strings_ignore_case_and_whitespace() -> None:
    verdict = score_case(expected(sku='FLOUR-001'), [observed(sku='  flour-001 ')])
    assert verdict is Verdict.CORRECT


def test_numbers_compare_as_numbers() -> None:
    verdict = score_case(expected(quantity=5), [observed(quantity=5.0)])
    assert verdict is Verdict.CORRECT


def test_numeric_string_is_not_a_number() -> None:
    verdict = score_case(expected(quantity=5), [observed(quantity='5')])
    assert verdict is Verdict.BAD_ARGUMENTS


def test_bool_is_not_a_number() -> None:
    verdict = score_case(expected(urgent=True), [observed(urgent=1)])
    assert verdict is Verdict.BAD_ARGUMENTS


def test_wrong_tool_name() -> None:
    verdict = score_case(expected(), [observed(name='list_open_purchase_orders')])
    assert verdict is Verdict.WRONG_TOOL


def test_wrong_tool_takes_precedence_over_arguments() -> None:
    verdict = score_case(expected(sku='A'), [observed(name='other_tool', sku='B')])
    assert verdict is Verdict.WRONG_TOOL


def test_extra_argument_is_bad_arguments() -> None:
    verdict = score_case(expected(sku='A'), [observed(sku='A', warehouse='W1')])
    assert verdict is Verdict.BAD_ARGUMENTS


def test_missing_argument_is_bad_arguments() -> None:
    verdict = score_case(expected(sku='A', warehouse='W1'), [observed(sku='A')])
    assert verdict is Verdict.BAD_ARGUMENTS


def test_nested_arguments_are_normalised() -> None:
    want = expected(
        name='draft_purchase_order',
        supplier='Acme Mills',
        lines=[{'sku': 'FLOUR-001', 'quantity': 100}],
    )
    got = observed(
        name='draft_purchase_order',
        supplier=' acme  mills',
        lines=[{'sku': 'flour-001', 'quantity': 100.0}],
    )
    assert score_case(want, [got]) is Verdict.CORRECT


def test_list_order_matters() -> None:
    want = expected(skus=['A', 'B'])
    assert score_case(want, [observed(skus=['B', 'A'])]) is Verdict.BAD_ARGUMENTS


def test_no_call_when_one_was_expected() -> None:
    assert score_case(expected(sku='A'), []) is Verdict.NO_CALL


def test_correct_abstain() -> None:
    assert score_case(None, []) is Verdict.CORRECT_ABSTAIN


def test_spurious_call_when_none_was_expected() -> None:
    assert score_case(None, [observed(sku='A')]) is Verdict.SPURIOUS_CALL


def test_only_the_first_call_is_scored() -> None:
    calls = [observed(sku='A'), observed(name='other_tool')]
    assert score_case(expected(sku='A'), calls) is Verdict.CORRECT


def test_accuracy_counts_correct_and_correct_abstain() -> None:
    verdicts = [
        Verdict.CORRECT,
        Verdict.CORRECT_ABSTAIN,
        Verdict.WRONG_TOOL,
        Verdict.BAD_ARGUMENTS,
    ]
    assert tool_call_accuracy(verdicts) == pytest.approx(0.5)


def test_accuracy_of_nothing_is_an_error() -> None:
    with pytest.raises(ValueError):
        tool_call_accuracy([])


def test_rejected_call_is_invalid_when_a_call_was_expected() -> None:
    assert score_case(expected(sku='A'), [], rejected=True) is Verdict.INVALID_CALL


def test_rejected_call_is_spurious_when_none_was_expected() -> None:
    assert score_case(None, [], rejected=True) is Verdict.SPURIOUS_CALL
