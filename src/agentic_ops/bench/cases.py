from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, model_validator

from agentic_ops.bench.scoring import ExpectedCall


class ToolDef(BaseModel):
    """A tool schema offered to the model; benchmarks never execute tools."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai(self) -> dict[str, Any]:
        """The OpenAI function-calling format that OpenAI-compatible providers accept."""
        return {
            'type': 'function',
            'function': {
                'name': self.name,
                'description': self.description,
                'parameters': self.parameters,
            },
        }

    def property_names(self) -> frozenset[str]:
        properties = self.parameters.get('properties')
        if not isinstance(properties, dict):
            return frozenset()
        return frozenset(str(key) for key in properties)

    def required_names(self) -> frozenset[str]:
        required = self.parameters.get('required')
        if not isinstance(required, list):
            return frozenset()
        return frozenset(str(name) for name in required)


class BenchCase(BaseModel):
    """One prompt and the call it requires; expected None means no tool should be called."""

    model_config = ConfigDict(frozen=True)

    id: str
    prompt: str
    expected: ExpectedCall | None


class CaseSuite(BaseModel):
    """A versioned set of tools and cases, checked for internal consistency at load."""

    model_config = ConfigDict(frozen=True)

    version: str
    tools: list[ToolDef]
    cases: list[BenchCase]

    @model_validator(mode='after')
    def _check_integrity(self) -> Self:
        tools = {tool.name: tool for tool in self.tools}
        if len(tools) != len(self.tools):
            raise ValueError('Duplicate tool name')
        seen: set[str] = set()
        for case in self.cases:
            if case.id in seen:
                raise ValueError(f'Duplicate case id {case.id!r}')
            seen.add(case.id)
            if case.expected is None:
                continue
            tool = tools.get(case.expected.name)
            if tool is None:
                raise ValueError(f'Unknown tool {case.expected.name!r} in case {case.id!r}')
            given = set(case.expected.args)
            extra = sorted(given - tool.property_names())
            if extra:
                names = ', '.join(extra)
                raise ValueError(f'Arguments {names} not in schema of {tool.name!r} ({case.id!r})')
            missing = sorted(tool.required_names() - given)
            if missing:
                names = ', '.join(missing)
                raise ValueError(
                    f'Missing required arguments {names} for {tool.name!r} ({case.id!r})'
                )
        return self


def load_suite(path: Path) -> CaseSuite:
    """Read and validate a case file; a malformed file fails loudly before any model call."""
    return CaseSuite.model_validate_json(path.read_text(encoding='utf-8'))
