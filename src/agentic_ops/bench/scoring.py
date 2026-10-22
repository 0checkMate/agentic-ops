from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict


class Verdict(StrEnum):
    """Outcome of one benchmark case; the six classes separate distinct failure modes."""

    CORRECT = 'correct'
    WRONG_TOOL = 'wrong_tool'
    BAD_ARGUMENTS = 'bad_arguments'
    NO_CALL = 'no_call'
    SPURIOUS_CALL = 'spurious_call'
    CORRECT_ABSTAIN = 'correct_abstain'
    INVALID_CALL = 'invalid_call'


class ExpectedCall(BaseModel):
    """The tool call a case requires: a tool name and its arguments."""

    model_config = ConfigDict(frozen=True)

    name: str
    args: dict[str, Any]


class ObservedCall(BaseModel):
    """A tool call a model actually made, already extracted from the provider response."""

    model_config = ConfigDict(frozen=True)

    name: str
    args: dict[str, Any]


_PASSING = frozenset({Verdict.CORRECT, Verdict.CORRECT_ABSTAIN})


def _fold(text: str) -> str:
    """Trim, collapse internal whitespace and casefold."""
    return ' '.join(text.split()).casefold()


def _equal(want: object, got: object) -> bool:
    """Strict structural equality with text folding; types are never coerced."""
    if isinstance(want, bool) or isinstance(got, bool):
        return isinstance(want, bool) and isinstance(got, bool) and want == got
    if isinstance(want, str) and isinstance(got, str):
        return _fold(want) == _fold(got)
    if isinstance(want, int | float) and isinstance(got, int | float):
        return want == got
    if isinstance(want, Mapping) and isinstance(got, Mapping):
        return want.keys() == got.keys() and all(_equal(want[key], got[key]) for key in want)
    if isinstance(want, list | tuple) and isinstance(got, list | tuple):
        return len(want) == len(got) and all(
            _equal(left, right) for left, right in zip(want, got, strict=True)
        )
    return want == got


def score_case(
    expected: ExpectedCall | None, observed: Sequence[ObservedCall], *, rejected: bool = False
) -> Verdict:
    """Classify one case; expected None means the model should answer without a tool.

    rejected means the provider refused the model's own tool call as schema-invalid. That
    is a failure of the model, scored by what the case wanted: a spurious call when no
    tool was expected, an invalid call otherwise. Only the first observed call is scored:
    these cases are single-step by design.
    """
    if rejected:
        return Verdict.SPURIOUS_CALL if expected is None else Verdict.INVALID_CALL
    if expected is None:
        return Verdict.SPURIOUS_CALL if observed else Verdict.CORRECT_ABSTAIN
    if not observed:
        return Verdict.NO_CALL
    first = observed[0]
    if first.name != expected.name:
        return Verdict.WRONG_TOOL
    return Verdict.CORRECT if _equal(expected.args, first.args) else Verdict.BAD_ARGUMENTS


def tool_call_accuracy(verdicts: Sequence[Verdict]) -> float:
    """Share of cases that were correct, abstentions included; empty input is an error."""
    if not verdicts:
        raise ValueError('Cannot compute accuracy of zero cases')
    return sum(verdict in _PASSING for verdict in verdicts) / len(verdicts)
