from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class SessionOpenRequest(BaseModel):
    telephony_call_id: str
    operator_id: str
    stir_shaken_attestation: str = "unknown"
    caller_phone: str | None = None


class SessionSummary(BaseModel):
    session_id: str
    current_state: str
    mode: str
    started_at: datetime
    operator_id: str
    ended_at: datetime | None = None
    disposition: str | None = None


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary]
