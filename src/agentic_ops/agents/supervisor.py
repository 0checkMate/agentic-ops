"""Supervisor graph: one routing LLM call per turn; specialists end the turn."""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

ROUTE_TOOL_NAME = 'route'
UNROUTABLE_REPLY = 'I cannot route that request to an available specialist.'


class AgentState(MessagesState):
    """Thread state. Extend here (never in nodes) when a field must survive turns."""


Specialist = Callable[[AgentState], Awaitable[dict[str, Any]]]


@tool(ROUTE_TOOL_NAME)
def route(agent: str) -> str:
    """Hand the request to one specialist agent, identified by name."""
    return agent  # never executed: the supervisor reads the call, it does not run it


def _system_prompt(names: Sequence[str]) -> SystemMessage:
    return SystemMessage(
        content=(
            'You are the operations supervisor. Route each request to exactly one '
            f'specialist by calling the {ROUTE_TOOL_NAME} tool, or answer directly if no '
            f'specialist applies. Available specialists: {", ".join(names)}.'
        )
    )


def _route_target(reply: AIMessage) -> str | None:
    for call in reply.tool_calls:
        if call['name'] == ROUTE_TOOL_NAME:
            agent = call['args'].get('agent')
            return agent if isinstance(agent, str) else ''
    return None


def build_supervisor_graph(
    model: BaseChatModel,
    specialists: Mapping[str, Specialist],
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[AgentState, Any, AgentState, AgentState]:
    routing_model = model.bind_tools([route])
    system = _system_prompt(list(specialists))

    async def supervisor(state: AgentState) -> Command[str]:
        reply = await routing_model.ainvoke([system, *state['messages']])
        if not isinstance(reply, AIMessage):
            raise TypeError(f'expected AIMessage, got {type(reply).__name__}')
        target = _route_target(reply)
        if target is None:
            # Direct answer. Rebuild the message so no stray tool_calls are persisted.
            return Command(goto=END, update={'messages': [AIMessage(content=reply.content)]})
        if target not in specialists:
            return Command(goto=END, update={'messages': [AIMessage(content=UNROUTABLE_REPLY)]})
        return Command(goto=target)  # the routing message is deliberately not stored

    builder = StateGraph(AgentState)
    builder.add_node('supervisor', supervisor)
    for name, node in specialists.items():
        builder.add_node(name, cast(Any, node))
        builder.add_edge(name, END)
    builder.add_edge(START, 'supervisor')
    return builder.compile(checkpointer=checkpointer)
