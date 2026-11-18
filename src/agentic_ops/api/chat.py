"""Chat route: thread-scoped conversation with the compiled agent graph."""

import logging
from typing import Any, Protocol

from fastapi import APIRouter, HTTPException
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

THREAD_ID_PATTERN = r'^[A-Za-z0-9_.:-]+$'
FAILURE_DETAIL = 'The agent service could not process the request.'


class ChatGraph(Protocol):
    """The only part of a compiled LangGraph graph the API depends on."""

    async def ainvoke(self, input: dict[str, Any], config: RunnableConfig, /) -> dict[str, Any]: ...


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    thread_id: str = Field(min_length=1, max_length=128, pattern=THREAD_ID_PATTERN)
    message: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    thread_id: str
    reply: str
    agent: str | None


def _text_of(content: object) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get('type') == 'text':
                text = block.get('text')
                if isinstance(text, str):
                    parts.append(text)
    return ''.join(parts)


def build_chat_router(graph: ChatGraph) -> APIRouter:
    router = APIRouter()

    @router.post('/chat')
    async def chat(request: ChatRequest) -> ChatResponse:
        config: RunnableConfig = {'configurable': {'thread_id': request.thread_id}}
        try:
            result = await graph.ainvoke({'messages': [HumanMessage(request.message)]}, config)
        except Exception:
            logger.exception('chat graph failed', extra={'thread_id': request.thread_id})
            raise HTTPException(status_code=502, detail=FAILURE_DETAIL) from None
        messages = result.get('messages')
        last = messages[-1] if isinstance(messages, list) and messages else None
        if not isinstance(last, AIMessage):
            logger.error('chat graph returned no assistant message')
            raise HTTPException(status_code=502, detail=FAILURE_DETAIL)
        return ChatResponse(
            thread_id=request.thread_id, reply=_text_of(last.content), agent=last.name
        )

    return router
