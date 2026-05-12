from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class SupervisorInterventionRequest(BaseModel):
    intervention_type: str
    reason_code: str
    notes: str = ""


class SupervisorInterventionResponse(BaseModel):
    session_id: str
    current_state: str
    intervention_recorded: bool = True


class AuditEventSchema(BaseModel):
    event_id: str
    event_type: str
    actor: str
    timestamp: datetime
    template_id: str | None = None
    before_after_hash: str
    chain_hash: str = ""


class AuditResponse(BaseModel):
    session_id: str
    events: list[AuditEventSchema]


class AuditVerifyResponse(BaseModel):
    valid: bool
    errors: list[str]
    events_verified: int
