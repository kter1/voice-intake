from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from voice_intake.api.deps import get_orchestrator, get_store, load_session_or_404
from voice_intake.api.schemas.session import (
    SessionListResponse,
    SessionOpenRequest,
    SessionSummary,
)
from voice_intake.db import SQLAuditStore
from voice_intake.models import StirShakenAttestation
from voice_intake.orchestrator import VoiceIntakeOrchestrator

router = APIRouter(tags=["sessions"])


# ── Appointment sub-resource schema ──────────────────────────────────────────

class AppointmentRequestSchema(BaseModel):
    appointment_id: str
    session_id: str
    reason_for_visit: str | None
    patient_name: str | None
    date_of_birth: str | None
    insurance_name: str | None
    insurance_network_status: str
    appointment_status: str
    scheduled_slot: str | None
    created_at: datetime
    updated_at: datetime


@router.post("/session/open", status_code=status.HTTP_201_CREATED, response_model=SessionSummary)
def open_session(
    body: SessionOpenRequest,
    store: SQLAuditStore = Depends(get_store),
    orchestrator: VoiceIntakeOrchestrator = Depends(get_orchestrator),
) -> SessionSummary:
    try:
        attestation = StirShakenAttestation(body.stir_shaken_attestation)
    except ValueError:
        attestation = StirShakenAttestation.UNKNOWN

    session = orchestrator.open_session(
        telephony_call_id=body.telephony_call_id,
        operator_id=body.operator_id,
        attestation=attestation,
    )
    orchestrator.start_after_opening(session)
    store.save_session(session, caller_phone=body.caller_phone)

    return SessionSummary(
        session_id=session.session_id,
        current_state=session.current_state.value,
        mode=session.mode.value,
        started_at=session.started_at,
        operator_id=session.operator_id,
    )


@router.get("/sessions", response_model=SessionListResponse)
def list_sessions(store: SQLAuditStore = Depends(get_store)) -> SessionListResponse:
    sessions = store.list_sessions()
    return SessionListResponse(
        sessions=[
            SessionSummary(
                session_id=s.session_id,
                current_state=s.current_state.value,
                mode=s.mode.value,
                started_at=s.started_at,
                operator_id=s.operator_id,
                ended_at=s.ended_at,
                disposition=s.disposition.value if s.disposition else None,
            )
            for s in sessions
        ]
    )


@router.get("/session/{session_id}", response_model=SessionSummary)
def get_session(session=Depends(load_session_or_404)) -> SessionSummary:
    return SessionSummary(
        session_id=session.session_id,
        current_state=session.current_state.value,
        mode=session.mode.value,
        started_at=session.started_at,
        operator_id=session.operator_id,
        ended_at=session.ended_at,
        disposition=session.disposition.value if session.disposition else None,
    )


@router.get(
    "/session/{session_id}/appointment",
    response_model=AppointmentRequestSchema,
)
def get_appointment(
    session_id: str,
    store: SQLAuditStore = Depends(get_store),
) -> AppointmentRequestSchema:
    """Return the structured appointment request for a demo session.

    Returns 404 if no appointment has been created yet for this session
    (i.e. the demo router hasn't committed any patch yet).
    """
    appt = store.get_appointment_request(session_id)
    if appt is None:
        raise HTTPException(
            status_code=404,
            detail=f"No appointment request found for session {session_id!r}",
        )
    return AppointmentRequestSchema(
        appointment_id=appt.appointment_id,
        session_id=appt.session_id,
        reason_for_visit=appt.reason_for_visit,
        patient_name=appt.patient_name,
        date_of_birth=appt.date_of_birth,
        insurance_name=appt.insurance_name,
        insurance_network_status=appt.insurance_network_status.value,
        appointment_status=appt.appointment_status.value,
        scheduled_slot=appt.scheduled_slot,
        created_at=appt.created_at,
        updated_at=appt.updated_at,
    )
