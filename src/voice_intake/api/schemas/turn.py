from __future__ import annotations

from pydantic import BaseModel, Field


class TurnRequest(BaseModel):
    transcript: str = Field(default="", max_length=2000)
    asr_confidence: float = 1.0
    partial_or_final: str = "final"
    audio_b64: str | None = Field(default=None, max_length=4_000_000)


class VoiceActionSchema(BaseModel):
    action_type: str
    template_id: str
    allowed_variables: dict[str, str]
    interruptible: bool
    timeout_ms: int
    # Guard-approved natural phrasing (demo profile only). Clients speak this
    # when present and fall back to rendering template_id otherwise.
    spoken_text: str | None = None


class RejectionSchema(BaseModel):
    reason: str


class TurnResponse(BaseModel):
    session_id: str
    turn_id: str
    current_state: str
    voice_action: VoiceActionSchema | None = None
    rejection: RejectionSchema | None = None
