from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from agentic_ops.db.models import AuditLog


async def record_event(
    session: AsyncSession,
    *,
    thread_id: str,
    event_type: str,
    actor: str,
    payload: dict[str, Any] | None = None,
) -> AuditLog:
    """Append one audit event to the caller's transaction.

    The caller owns the transaction: this function flushes but never commits, so the
    audit row commits or rolls back atomically with the action it describes. Payloads
    must be JSON-safe; pass model_dump(mode='json') for pydantic models.
    """
    event = AuditLog(
        thread_id=thread_id,
        event_type=event_type,
        actor=actor,
        payload=payload if payload is not None else {},
    )
    session.add(event)
    await session.flush()
    return event
