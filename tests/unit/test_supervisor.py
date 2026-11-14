"""Supervisor routing: handoff by tool call, direct answer, unknown agent, persistence."""

from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from agentic_ops.agents.supervisor import (
    ROUTE_TOOL_NAME,
    UNROUTABLE_REPLY,
    AgentState,
    Specialist,
    build_supervisor_graph,
)

CONFIG: RunnableConfig = {'configurable': {'thread_id': 'thread-1'}}


class ScriptedChatModel(BaseChatModel):
    """Replays scripted AIMessages in order and records what it was shown."""

    script: list[AIMessage]
    seen: list[list[BaseMessage]] = []
    cursor: int = 0

    @property
    def _llm_type(self) -> str:
        return 'scripted'

    def bind_tools(
        self, tools: Sequence[Any], **kwargs: Any
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen.append(list(messages))
        reply = self.script[self.cursor]
        self.cursor += 1
        return ChatResult(generations=[ChatGeneration(message=reply)])


def route_call(agent: str) -> AIMessage:
    return AIMessage(
        content='',
        tool_calls=[
            {'name': ROUTE_TOOL_NAME, 'args': {'agent': agent}, 'id': 'call_1', 'type': 'tool_call'}
        ],
    )


def recording_specialist(name: str, calls: list[str]) -> Specialist:
    async def node(state: AgentState) -> dict[str, Any]:
        calls.append(name)
        return {'messages': [AIMessage(content=f'{name} done', name=name)]}

    return node


async def test_routes_to_specialist_and_ends_turn() -> None:
    calls: list[str] = []
    model = ScriptedChatModel(script=[route_call('procurement')])
    graph = build_supervisor_graph(
        model,
        {'procurement': recording_specialist('procurement', calls)},
        checkpointer=InMemorySaver(),
    )
    result = await graph.ainvoke({'messages': [HumanMessage('Which supplier for flour?')]}, CONFIG)
    assert calls == ['procurement']
    assert result['messages'][-1].content == 'procurement done'
    assert model.cursor == 1  # one LLM call: specialists end the turn


async def test_answers_directly_without_tool_call() -> None:
    calls: list[str] = []
    model = ScriptedChatModel(script=[AIMessage(content='Hello, how can I help?')])
    graph = build_supervisor_graph(
        model,
        {'procurement': recording_specialist('procurement', calls)},
        checkpointer=InMemorySaver(),
    )
    result = await graph.ainvoke({'messages': [HumanMessage('hi')]}, CONFIG)
    assert calls == []
    assert result['messages'][-1].content == 'Hello, how can I help?'


async def test_unknown_agent_is_not_followed() -> None:
    calls: list[str] = []
    model = ScriptedChatModel(script=[route_call('payroll')])
    graph = build_supervisor_graph(
        model,
        {'procurement': recording_specialist('procurement', calls)},
        checkpointer=InMemorySaver(),
    )
    result = await graph.ainvoke({'messages': [HumanMessage('pay everyone')]}, CONFIG)
    assert calls == []
    assert result['messages'][-1].content == UNROUTABLE_REPLY


async def test_routing_message_is_not_persisted() -> None:
    model = ScriptedChatModel(script=[route_call('procurement')])
    graph = build_supervisor_graph(
        model,
        {'procurement': recording_specialist('procurement', [])},
        checkpointer=InMemorySaver(),
    )
    result = await graph.ainvoke({'messages': [HumanMessage('reorder flour')]}, CONFIG)
    assert not any(isinstance(m, AIMessage) and m.tool_calls for m in result['messages'])


async def test_prompt_lists_every_specialist() -> None:
    model = ScriptedChatModel(script=[AIMessage(content='ok')])
    specialists = {
        'procurement': recording_specialist('procurement', []),
        'quality': recording_specialist('quality', []),
    }
    graph = build_supervisor_graph(model, specialists, checkpointer=InMemorySaver())
    await graph.ainvoke({'messages': [HumanMessage('hi')]}, CONFIG)
    first = model.seen[0][0]
    assert isinstance(first, SystemMessage)
    assert 'procurement' in str(first.content)
    assert 'quality' in str(first.content)


async def test_thread_history_persists_across_turns() -> None:
    model = ScriptedChatModel(script=[route_call('procurement'), AIMessage(content='ok')])
    graph = build_supervisor_graph(
        model,
        {'procurement': recording_specialist('procurement', [])},
        checkpointer=InMemorySaver(),
    )
    await graph.ainvoke({'messages': [HumanMessage('first question')]}, CONFIG)
    await graph.ainvoke({'messages': [HumanMessage('second question')]}, CONFIG)
    assert any(isinstance(m, HumanMessage) and m.content == 'first question' for m in model.seen[1])
