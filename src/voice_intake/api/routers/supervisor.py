from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from voice_intake.api.deps import get_orchestrator, get_store, load_session_or_404
from voice_intake.api.schemas.supervisor import (
    SupervisorInterventionRequest,
    SupervisorInterventionResponse,
)
from voice_intake.db import SQLAuditStore
from voice_intake.models import CallSession, SupervisorInterventionType
from voice_intake.orchestrator import VoiceIntakeOrchestrator

router = APIRouter(tags=["supervisor"])


@router.post("/session/{session_id}/supervisor/intervene", response_model=SupervisorInterventionResponse)
def supervisor_intervene(
    session_id: str,
    body: SupervisorInterventionRequest,
    store: SQLAuditStore = Depends(get_store),
    orchestrator: VoiceIntakeOrchestrator = Depends(get_orchestrator),
    session: CallSession = Depends(load_session_or_404),
) -> SupervisorInterventionResponse:
    try:
        intervention_type = SupervisorInterventionType(body.intervention_type)
    except ValueError:
        valid = [t.value for t in SupervisorInterventionType]
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid intervention_type {body.intervention_type!r}. Valid: {valid}",
        )

    orchestrator.record_supervisor_intervention(
        session, intervention_type, body.reason_code, body.notes
    )
    store.save_session(session)

    return SupervisorInterventionResponse(
        session_id=session_id,
        current_state=session.current_state.value,
    )
