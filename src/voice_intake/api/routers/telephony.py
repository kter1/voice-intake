"""Telephony HTTP endpoint: Twilio inbound voice webhook.

The WebSocket endpoint ``/ws/telephony/{session_id}`` is registered directly
on the FastAPI app in ``api/app.py`` (same pattern as ``/ws/supervisors``).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from voice_intake.api.deps import get_orchestrator, get_settings, get_store
from voice_intake.api.security import make_stream_token
from voice_intake.config import Settings
from voice_intake.db import SQLAuditStore
from voice_intake.models import StirShakenAttestation
from voice_intake.orchestrator import VoiceIntakeOrchestrator
from voice_intake.telephony.twilio_adapter import (
    build_stream_twiml,
    parse_stir_verstat,
    validate_twilio_signature,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/telephony", tags=["telephony"])


@router.post("/twilio/voice")
async def twilio_voice_webhook(
    request: Request,
    settings: Settings = Depends(get_settings),
    store: SQLAuditStore = Depends(get_store),
    orchestrator: VoiceIntakeOrchestrator = Depends(get_orchestrator),
) -> Response:
    """Handle an inbound Twilio call.

    Validates the Twilio request signature, opens a voice-intake session,
    and returns TwiML that connects the call to our Media Stream WebSocket.

    The ``mock_twilio_validate`` setting skips signature checking (for tests
    and local development).
    """
    # --- Parse form body -------------------------------------------------------
    form = await request.form()
    params: dict[str, str] = {k: str(v) for k, v in form.items()}

    # --- Signature validation --------------------------------------------------
    if not settings.mock_twilio_validate:
        signature = request.headers.get("X-Twilio-Signature", "")
        full_url = str(request.url)
        if not validate_twilio_signature(
            settings.twilio_auth_token, signature, full_url, params
        ):
            logger.warning(
                "Twilio signature validation failed url=%s", full_url
            )
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    # --- STIR/SHAKEN -----------------------------------------------------------
    attestation = parse_stir_verstat(params.get("StirVerstat"))

    # --- Caller / call metadata -----------------------------------------------
    caller_phone = params.get("From", "unknown")
    call_sid = params.get("CallSid", "unknown")
    operator_id = params.get("To", "twilio")   # called number = practice line

    # --- Open intake session --------------------------------------------------
    session = orchestrator.open_session(
        telephony_call_id=call_sid,
        operator_id=operator_id,
        attestation=attestation,
    )
    orchestrator.start_after_opening(session)
    store.save_session(session, caller_phone=caller_phone)

    logger.info(
        "Telephony session opened session_id=%s call_sid=%s attestation=%s",
        session.session_id,
        call_sid,
        attestation.value,
    )

    # --- Build WebSocket stream URL -------------------------------------------
    # Derive ws(s):// base from the incoming request's scheme + host
    base = str(request.base_url).rstrip("/")
    ws_base = base.replace("https://", "wss://").replace("http://", "ws://")
    token = make_stream_token(session.session_id, request.app.state.stream_secret)
    stream_url = f"{ws_base}/ws/telephony/{session.session_id}?token={token}"

    twiml = build_stream_twiml(stream_url)
    return Response(content=twiml, media_type="text/xml")
