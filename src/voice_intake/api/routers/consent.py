from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from voice_intake.api.deps import get_orchestrator, get_store, load_session_or_404
from voice_intake.api.schemas.consent import ConsentRequest, ConsentResponse
from voice_intake.db import SQLAuditStore
from voice_intake.models import CallSession, ConsentType
from voice_intake.orchestrator import VoiceIntakeOrchestrator

router = APIRouter(tags=["consent"])


@router.post("/session/{session_id}/consent", response_model=ConsentResponse)
def capture_consent(
    session_id: str,
    body: ConsentRequest,
    store: SQLAuditStore = Depends(get_store),
    orchestrator: VoiceIntakeOrchestrator = Depends(get_orchestrator),
    session: CallSession = Depends(load_session_or_404),
) -> ConsentResponse:
    try:
        consent_type = ConsentType(body.consent_type)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid consent_type {body.consent_type!r}. "
                   "Use 'recording_consent' or 'ai_assistance_consent'.",
        )

    orchestrator.capture_consent(session, consent_type, body.granted)
    store.save_session(session)

    # Retrieve storage_effect from the most recent consent artifact
    artifacts = store.consents_for_session(session_id)
    matching = [a for a in artifacts if a.consent_type == consent_type]
    storage_effect = matching[-1].storage_effect if matching else ""

    return ConsentResponse(
        session_id=session_id,
        next_state=session.current_state.value,
        mode=session.mode.value,
        storage_effect=storage_effect,
    )
