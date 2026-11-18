"""Composition root: the only module that turns settings into live objects."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.types import Lifespan

from agentic_ops.agents.audit import AuditCallbackHandler, AuditSink, PostgresAuditSink
from agentic_ops.agents.procurement import agent_as_specialist, build_procurement_agent
from agentic_ops.agents.supervisor import build_supervisor_graph
from agentic_ops.api.app import create_app, sql_db_check
from agentic_ops.api.audited import AuditedGraph
from agentic_ops.api.lazy import LazyGraph
from agentic_ops.config import Settings, get_settings
from agentic_ops.db.checkpoint import open_checkpointer
from agentic_ops.db.session import get_sessionmaker
from agentic_ops.erp.mock.demo import demo_erp
from agentic_ops.erp.port import ErpPort
from agentic_ops.gateway.chat_model import GatewayChatModel, provider_chat_call
from agentic_ops.gateway.defaults import default_limits, default_registry
from agentic_ops.gateway.ratelimit import RateLimiter
from agentic_ops.gateway.router import Gateway


def gateway_chat_model(settings: Settings) -> GatewayChatModel:
    """The production chat model: routing, rate limits and fallback on every request."""
    gateway = Gateway(default_registry(), RateLimiter(default_limits()))
    return GatewayChatModel(gateway=gateway, call=provider_chat_call(settings))


def build_chat_graph(
    model: BaseChatModel,
    erp: ErpPort,
    *,
    checkpointer: BaseCheckpointSaver[Any],
    audit: AuditSink,
) -> AuditedGraph:
    """Assemble supervisor and specialists; no settings, no I/O."""
    specialists = {'procurement': agent_as_specialist(build_procurement_agent(model, erp))}
    graph = build_supervisor_graph(model, specialists, checkpointer=checkpointer, audit=audit)
    return AuditedGraph(graph, AuditCallbackHandler(audit))


def live_lifespan(
    settings: Settings,
    lazy: LazyGraph,
    session_factory: async_sessionmaker[AsyncSession],
    *,
    model: BaseChatModel | None = None,
) -> Lifespan[FastAPI]:
    """Open the checkpointer, build the graph, serve, then close in reverse order."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        chat_model = model if model is not None else gateway_chat_model(settings)
        sink = PostgresAuditSink(session_factory)
        async with open_checkpointer(settings.database_url.get_secret_value()) as saver:
            lazy.set(build_chat_graph(chat_model, demo_erp(), checkpointer=saver, audit=sink))
            try:
                yield
            finally:
                lazy.clear()

    return lifespan


def create_live_app() -> FastAPI:
    """Entry point: uvicorn agentic_ops.api.main:create_live_app --factory."""
    settings = get_settings()
    sessions = get_sessionmaker()
    lazy = LazyGraph()
    return create_app(
        sql_db_check(sessions),
        graph=lazy,
        lifespan=live_lifespan(settings, lazy, sessions),
    )
