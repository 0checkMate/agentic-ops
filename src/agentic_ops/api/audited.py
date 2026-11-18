"""Graph adapter that attaches the audit callback handler to every run."""

from typing import Any, Protocol

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig

from agentic_ops.agents.supervisor import AgentState


class AgentGraph(Protocol):
    """The part of a compiled supervisor graph this adapter calls."""

    async def ainvoke(self, input: AgentState, config: RunnableConfig, /) -> Any: ...


class AuditedGraph:
    """Satisfies ChatGraph; the HTTP layer never learns that auditing exists."""

    def __init__(self, graph: AgentGraph, handler: BaseCallbackHandler) -> None:
        self._graph = graph
        self._handler = handler

    async def ainvoke(self, input: dict[str, Any], config: RunnableConfig, /) -> dict[str, Any]:
        existing = config.get('callbacks')
        if existing is None:
            callbacks: list[BaseCallbackHandler] = []
        elif isinstance(existing, list):
            callbacks = list(existing)
        else:
            raise TypeError('AuditedGraph supports only a list of callbacks in the config')
        callbacks.append(self._handler)
        audited = config.copy()
        audited['callbacks'] = callbacks
        result = await self._graph.ainvoke(AgentState(messages=input['messages']), audited)
        if not isinstance(result, dict):
            raise TypeError(f'expected a state dict, got {type(result).__name__}')
        return result
