from __future__ import annotations

from pydantic import BaseModel


class ConsentRequest(BaseModel):
    consent_type: str  # "recording_consent" or "ai_assistance_consent"
    granted: bool


class ConsentResponse(BaseModel):
    session_id: str
    next_state: str
    mode: str
    storage_effect: str
