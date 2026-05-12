from __future__ import annotations

from fastapi import APIRouter, Depends

from voice_intake.api.deps import get_store, load_session_or_404
from voice_intake.api.schemas.supervisor import AuditEventSchema, AuditResponse, AuditVerifyResponse
from voice_intake.audit_verify import verify_chain
from voice_intake.db import SQLAuditStore
from voice_intake.models import AuditEventType, CallSession

router = APIRouter(tags=["audit"])


@router.get("/session/{session_id}/audit", response_model=AuditResponse)
def get_audit(
    session_id: str,
    store: SQLAuditStore = Depends(get_store),
    _session: CallSession = Depends(load_session_or_404),
) -> AuditResponse:
    events = store.audit_events_for_session(session_id)
    return AuditResponse(
        session_id=session_id,
        events=[
            AuditEventSchema(
                event_id=e.event_id,
                event_type=(
                    e.event_type.value
                    if isinstance(e.event_type, AuditEventType)
                    else str(e.event_type)
                ),
                actor=e.actor.value,
                timestamp=e.timestamp,
                template_id=e.template_id,
                before_after_hash=e.before_after_hash,
                chain_hash=e.chain_hash,
                # details omitted intentionally - prevents PHI leakage in HTTP responses
            )
            for e in events
        ],
    )


@router.get("/session/{session_id}/audit/verify", response_model=AuditVerifyResponse)
def verify_audit(
    session_id: str,
    store: SQLAuditStore = Depends(get_store),
    _session: CallSession = Depends(load_session_or_404),
) -> AuditVerifyResponse:
    events = store.audit_events_for_session(session_id)
    errors = verify_chain(events, store.audit_secret)
    verified_count = len([event for event in events if event.chain_index is not None and event.chain_hash])
    return AuditVerifyResponse(
        valid=not errors,
        errors=errors,
        events_verified=verified_count,
    )
