from __future__ import annotations

from uuid import uuid4

from voice_intake.asr.interface import ASRResult
from voice_intake.models import Speaker, VoiceTurn, utc_now


def _new_id() -> str:
    return str(uuid4())


class MockASRAdapter:
    """Async ASRInterface-compatible adapter for test environments."""

    async def transcribe(self, audio_bytes: bytes, language: str = "en") -> ASRResult:
        # Decode bytes as UTF-8 text (test callers pass pre-transcribed text as bytes)
        text = audio_bytes.decode("utf-8", errors="replace")
        return ASRResult(transcript=text, confidence=1.0, is_final=True, language=language)


class MockASR:
    def transcribe(self, text: str, turn_id: str | None = None) -> VoiceTurn:
        ts = utc_now()
        return VoiceTurn(
            turn_id=turn_id or _new_id(),
            speaker=Speaker.CALLER,
            audio_ref=None,
            retention_policy_id="mock",
            transcript=text,
            partial_or_final="final",
            asr_confidence=1.0,
            barge_in=False,
            start_ts=ts,
            end_ts=ts,
        )
