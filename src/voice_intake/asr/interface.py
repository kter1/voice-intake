from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class ASRResult:
    transcript: str
    confidence: float
    is_final: bool
    language: str = "en"


@runtime_checkable
class ASRInterface(Protocol):
    async def transcribe(self, audio_bytes: bytes, language: str = "en") -> ASRResult: ...
