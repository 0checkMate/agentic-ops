from collections.abc import Sequence
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from agentic_ops.db.audit import record_event
from agentic_ops.db.models import AuditLog
from agentic_ops.gateway.errors import GatewayExhausted
from agentic_ops.gateway.models import Alias, ModelSpec, Skipped
from agentic_ops.gateway.router import RouteOutcome

MODEL_CALL = 'model_call'
MODEL_CALL_FAILED = 'model_call_failed'


def _build_payload(
    *,
    alias: Alias,
    spec: ModelSpec | None,
    skipped: Sequence[Skipped],
    estimated_tokens: int,
    latency_ms: float,
    cache_hit: bool,
) -> dict[str, Any]:
    return {
        'alias': alias.value,
        'provider': spec.provider if spec else None,
        'model_id': spec.model_id if spec else None,
        'fell_back': bool(skipped),
        'skipped': [item.model_dump(mode='json') for item in skipped],
        'estimated_tokens': estimated_tokens,
        'cache_hit': cache_hit,
        'latency_ms': latency_ms,
    }


def model_call_payload(
    *,
    alias: Alias,
    outcome: RouteOutcome[Any] | None,
    estimated_tokens: int,
    latency_ms: float,
) -> dict[str, Any]:
    """Routing metadata for one served call; outcome None means the cache answered.

    Prompts and responses are deliberately excluded: they may hold business data and
    would consume the database quota. Content belongs in LangSmith traces.
    """
    return _build_payload(
        alias=alias,
        spec=outcome.spec if outcome else None,
        skipped=outcome.skipped if outcome else (),
        estimated_tokens=estimated_tokens,
        latency_ms=latency_ms,
        cache_hit=outcome is None,
    )


def exhausted_payload(
    *,
    alias: Alias,
    error: GatewayExhausted,
    estimated_tokens: int,
    latency_ms: float,
) -> dict[str, Any]:
    """Routing metadata for a call that no route could serve."""
    return _build_payload(
        alias=alias,
        spec=None,
        skipped=error.skipped,
        estimated_tokens=estimated_tokens,
        latency_ms=latency_ms,
        cache_hit=False,
    )


async def record_model_call(
    session: AsyncSession,
    *,
    thread_id: str,
    actor: str,
    alias: Alias,
    outcome: RouteOutcome[Any] | None,
    estimated_tokens: int,
    latency_ms: float,
) -> AuditLog:
    """Append a model_call event to the caller's transaction."""
    payload = model_call_payload(
        alias=alias,
        outcome=outcome,
        estimated_tokens=estimated_tokens,
        latency_ms=latency_ms,
    )
    return await record_event(
        session, thread_id=thread_id, event_type=MODEL_CALL, actor=actor, payload=payload
    )


async def record_model_failure(
    session: AsyncSession,
    *,
    thread_id: str,
    actor: str,
    alias: Alias,
    error: GatewayExhausted,
    estimated_tokens: int,
    latency_ms: float,
) -> AuditLog:
    """Append a model_call_failed event to the caller's transaction."""
    payload = exhausted_payload(
        alias=alias,
        error=error,
        estimated_tokens=estimated_tokens,
        latency_ms=latency_ms,
    )
    return await record_event(
        session,
        thread_id=thread_id,
        event_type=MODEL_CALL_FAILED,
        actor=actor,
        payload=payload,
    )
