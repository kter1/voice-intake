"""GET /capabilities - advertise which backend services are configured.

The frontend reads this to show informational hints and prepare for deferred
backend-ASR mode.  No authentication required (the endpoint reveals no secrets).
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(tags=["capabilities"])


class CapabilitiesResponse(BaseModel):
    mock_asr: bool
    mock_llm: bool
    mock_rag: bool
    demo_router: bool
    backend_asr_configured: bool
    voice_modes: list[str]


@router.get("/capabilities", response_model=CapabilitiesResponse)
async def get_capabilities(request: Request) -> CapabilitiesResponse:
    settings = request.app.state.settings
    mock_asr: bool = getattr(settings, "mock_asr", True)
    mock_llm: bool = getattr(settings, "mock_llm", True)
    mock_rag: bool = getattr(settings, "mock_rag", True)
    demo_router_flag: bool = getattr(settings, "demo_router", False)
    deepgram_key: str = getattr(settings, "deepgram_api_key", "")

    backend_asr_configured = (not mock_asr) and bool(deepgram_key)

    voice_modes: list[str] = []
    if backend_asr_configured:
        voice_modes.append("backend_asr")
    voice_modes.extend(["browser_speech", "text"])

    return CapabilitiesResponse(
        mock_asr=mock_asr,
        mock_llm=mock_llm,
        mock_rag=mock_rag,
        demo_router=demo_router_flag,
        backend_asr_configured=backend_asr_configured,
        voice_modes=voice_modes,
    )
