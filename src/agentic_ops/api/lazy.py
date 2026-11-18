"""Late-bound graph: the router is built at import time, the graph at startup."""

from typing import Any

from langchain_core.runnables import RunnableConfig

from agentic_ops.api.chat import ChatGraph


class GraphNotReady(RuntimeError):
    """Raised when a chat arrives before the lifespan has built the graph."""


class LazyGraph:
    def __init__(self) -> None:
        self._graph: ChatGraph | None = None

    def set(self, graph: ChatGraph) -> None:
        self._graph = graph

    def clear(self) -> None:
        self._graph = None

    async def ainvoke(self, input: dict[str, Any], config: RunnableConfig, /) -> dict[str, Any]:
        graph = self._graph
        if graph is None:
            raise GraphNotReady('The agent graph has not been built yet.')
        return await graph.ainvoke(input, config)
