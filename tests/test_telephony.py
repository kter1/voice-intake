"""Phase 8: Telephony adapter tests.

Tests cover:
- STIR/SHAKEN verstat parsing
- Twilio signature validation helper
- TwiML builders
- POST /telephony/twilio/voice webhook (mock_twilio_validate=True)
- /ws/telephony/{session_id} Media Stream WebSocket in mock-ASR mode
"""

from __future__ import annotations

import json
import os

import pytest
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("MOCK_LLM", "true")
os.environ.setdefault("MOCK_ASR", "true")
os.environ.setdefault("MOCK_TWILIO_VALIDATE", "true")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")


def _mock_embed(texts: list[str]) -> list[list[float]]:
    return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

# ---------------------------------------------------------------------------
# STIR/SHAKEN helpers
# ---------------------------------------------------------------------------


class TestParseStirVerstat:
    def test_a_attestation(self):
        from voice_intake.telephony.twilio_adapter import parse_stir_verstat
        from voice_intake.models import StirShakenAttestation

        assert parse_stir_verstat("TN-Validation-Passed-A") == StirShakenAttestation.A

    def test_b_attestation(self):
        from voice_intake.telephony.twilio_adapter import parse_stir_verstat
        from voice_intake.models import StirShakenAttestation

        assert parse_stir_verstat("TN-Validation-Passed-B") == StirShakenAttestation.B

    def test_c_attestation(self):
        from voice_intake.telephony.twilio_adapter import parse_stir_verstat
        from voice_intake.models import StirShakenAttestation

        assert parse_stir_verstat("TN-Validation-Passed-C") == StirShakenAttestation.C

    def test_none_returns_unknown(self):
        from voice_intake.telephony.twilio_adapter import parse_stir_verstat
        from voice_intake.models import StirShakenAttestation

        assert parse_stir_verstat(None) == StirShakenAttestation.UNKNOWN

    def test_empty_string_returns_unknown(self):
        from voice_intake.telephony.twilio_adapter import parse_stir_verstat
        from voice_intake.models import StirShakenAttestation

        assert parse_stir_verstat("") == StirShakenAttestation.UNKNOWN

    def test_unrecognised_returns_unknown(self):
        from voice_intake.telephony.twilio_adapter import parse_stir_verstat
        from voice_intake.models import StirShakenAttestation

        assert parse_stir_verstat("TN-Validation-Failed") == StirShakenAttestation.UNKNOWN


# ---------------------------------------------------------------------------
# Signature validation helper
# ---------------------------------------------------------------------------


class TestValidateTwilioSignature:
    """validate_twilio_signature wraps twilio.request_validator.RequestValidator."""

    def test_valid_signature(self):
        """Real Twilio signature check using known token + URL + params."""
        from voice_intake.telephony.twilio_adapter import validate_twilio_signature

        # Use twilio SDK to generate a valid signature, then verify it
        from twilio.request_validator import RequestValidator  # type: ignore[import]

        token = "some_auth_token_123"
        url = "https://example.com/telephony/twilio/voice"
        params = {"CallSid": "CA123", "From": "+15551234567"}
        sig = RequestValidator(token).compute_signature(url, params)
        assert validate_twilio_signature(token, sig, url, params) is True

    def test_invalid_signature_returns_false(self):
        from voice_intake.telephony.twilio_adapter import validate_twilio_signature

        assert (
            validate_twilio_signature(
                "token", "badsig", "https://example.com/telephony/twilio/voice", {}
            )
            is False
        )


# ---------------------------------------------------------------------------
# TwiML builders
# ---------------------------------------------------------------------------


class TestTwiMLBuilders:
    def test_stream_twiml_contains_url(self):
        from voice_intake.telephony.twilio_adapter import build_stream_twiml

        twiml = build_stream_twiml("wss://example.com/ws/telephony/sess-1")
        assert "wss://example.com/ws/telephony/sess-1" in twiml
        assert "<Stream" in twiml
        assert "<Connect>" in twiml

    def test_say_twiml_contains_text(self):
        from voice_intake.telephony.twilio_adapter import build_say_twiml

        twiml = build_say_twiml("Hello, how can I help you today?")
        assert "Hello, how can I help you today?" in twiml
        assert "<Say" in twiml

    def test_say_twiml_escapes_ampersand(self):
        from voice_intake.telephony.twilio_adapter import build_say_twiml

        twiml = build_say_twiml("This & that")
        assert "&amp;" in twiml
        assert "&" in twiml  # the &amp; itself contains &

    def test_dial_twiml_contains_queue(self):
        from voice_intake.telephony.twilio_adapter import build_dial_twiml

        twiml = build_dial_twiml("my-queue")
        assert "my-queue" in twiml
        assert "<Queue>" in twiml
        assert "<Dial>" in twiml

    def test_dial_twiml_default_queue(self):
        from voice_intake.telephony.twilio_adapter import build_dial_twiml

        twiml = build_dial_twiml()
        assert "human-agent-queue" in twiml


# ---------------------------------------------------------------------------
# Inbound webhook: POST /telephony/twilio/voice
# ---------------------------------------------------------------------------


@pytest.fixture()
def telephony_client(tmp_path):
    """Async HTTPX client pointing at a fresh in-memory app instance."""
    from sqlalchemy.pool import StaticPool
    from voice_intake.api.app import create_app
    from voice_intake.db import get_engine, create_tables, make_session_factory
    from voice_intake.config import Settings

    settings = Settings(
        database_url="sqlite:///:memory:",
        mock_twilio_validate=True,
        mock_llm=True,
        mock_asr=True,
        audit_hmac_secret="test-audit-secret",
        stream_auth_secret="test-stream-secret",
    )
    app = create_app(settings=settings)
    app.state.embed_fn = _mock_embed

    from sqlalchemy import create_engine
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_tables(engine)
    app.state.session_factory = make_session_factory(engine)

    import chromadb  # type: ignore[import]
    app.state.chroma_client = chromadb.EphemeralClient()

    return app


@pytest.mark.asyncio
async def test_voice_webhook_creates_session(telephony_client):
    """POST /telephony/twilio/voice opens a session and returns TwiML stream."""
    app = telephony_client
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        resp = await client.post(
            "/telephony/twilio/voice",
            data={
                "CallSid": "CA_test_001",
                "From": "+15551234567",
                "To": "+18005551234",
                "StirVerstat": "TN-Validation-Passed-A",
            },
        )

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "text/xml; charset=utf-8"
    body = resp.text
    assert "<Stream" in body
    assert "ws://testserver/ws/telephony/" in body
    assert "token=" in body


@pytest.mark.asyncio
async def test_voice_webhook_stir_a_stored(telephony_client):
    """A-level attestation is persisted with the session."""
    from voice_intake.db import SQLAuditStore
    from voice_intake.models import StirShakenAttestation

    app = telephony_client
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        resp = await client.post(
            "/telephony/twilio/voice",
            data={
                "CallSid": "CA_test_002",
                "From": "+15551234567",
                "To": "+18005551234",
                "StirVerstat": "TN-Validation-Passed-A",
            },
        )
    assert resp.status_code == 200

    # Extract session_id from the stream URL in TwiML
    import re
    match = re.search(r"/ws/telephony/([\w-]+)", resp.text)
    assert match, f"No session ID in TwiML: {resp.text}"
    session_id = match.group(1)

    store = SQLAuditStore(app.state.session_factory, hmac_secret="test-audit-secret")
    session = store.load_session(session_id)
    assert session is not None
    assert session.stir_shaken_attestation == StirShakenAttestation.A


@pytest.mark.asyncio
async def test_voice_webhook_unknown_stir(telephony_client):
    """Missing StirVerstat falls back to UNKNOWN attestation."""
    from voice_intake.db import SQLAuditStore
    from voice_intake.models import StirShakenAttestation

    app = telephony_client
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        resp = await client.post(
            "/telephony/twilio/voice",
            data={
                "CallSid": "CA_test_003",
                "From": "+15559999999",
                "To": "+18005551234",
                # No StirVerstat
            },
        )
    assert resp.status_code == 200

    import re
    match = re.search(r"/ws/telephony/([\w-]+)", resp.text)
    session_id = match.group(1)  # type: ignore[union-attr]

    store = SQLAuditStore(app.state.session_factory, hmac_secret="test-audit-secret")
    session = store.load_session(session_id)
    assert session is not None
    assert session.stir_shaken_attestation == StirShakenAttestation.UNKNOWN


# ---------------------------------------------------------------------------
# Media Stream WebSocket
# ---------------------------------------------------------------------------


def _stream_messages(session_id: str, transcript_text: str) -> list[str]:
    """Build the JSON messages a Twilio Media Stream WS would send."""
    import base64

    audio_b64 = base64.b64encode(transcript_text.encode()).decode()
    return [
        json.dumps({"event": "connected", "protocol": "Call", "version": "1.0"}),
        json.dumps({
            "event": "start",
            "streamSid": "MZ_stream_001",
            "start": {
                "streamSid": "MZ_stream_001",
                "callSid": "CA_ws_001",
                "accountSid": "AC_test",
                "tracks": ["inbound"],
                "mediaFormat": {
                    "encoding": "audio/x-mulaw",
                    "sampleRate": 8000,
                    "channels": 1,
                },
            },
        }),
        json.dumps({
            "event": "media",
            "streamSid": "MZ_stream_001",
            "media": {
                "track": "inbound",
                "chunk": "1",
                "timestamp": "5",
                "payload": audio_b64,
            },
        }),
        json.dumps({
            "event": "stop",
            "streamSid": "MZ_stream_001",
            "stop": {"callSid": "CA_ws_001", "accountSid": "AC_test"},
        }),
    ]


@pytest.mark.asyncio
async def test_media_stream_processes_transcript(telephony_client):
    """Mock media stream → transcript → turn recorded in DB."""
    from httpx import AsyncClient as HC

    app = telephony_client

    # First open a session via the webhook
    async with HC(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        resp = await client.post(
            "/telephony/twilio/voice",
            data={"CallSid": "CA_ws_001", "From": "+15550001111", "To": "+18001234567"},
        )
    assert resp.status_code == 200

    import re
    match = re.search(r'url="(ws://testserver/ws/telephony/[^"]+)"', resp.text)
    assert match
    stream_url = match.group(1)
    session_match = re.search(r"/ws/telephony/([\w-]+)", stream_url)
    assert session_match
    session_id = session_match.group(1)

    # Connect to the Media Stream WebSocket and replay messages
    from starlette.testclient import TestClient
    from starlette.websockets import WebSocket as StarletteWS

    # Use Starlette's TestClient WebSocket support
    client = TestClient(app, raise_server_exceptions=False)
    messages = _stream_messages(session_id, "Hello I need an appointment")

    ws_path = stream_url.removeprefix("ws://testserver")
    with client.websocket_connect(ws_path) as ws:
        for msg in messages:
            ws.send_text(msg)
        # Give the handler time to process
        import time
        time.sleep(0.2)

    # Verify a turn was recorded
    from voice_intake.db import SQLAuditStore
    store = SQLAuditStore(app.state.session_factory, hmac_secret="test-audit-secret")
    turns = store.turns_for_session(session_id)
    assert len(turns) >= 1
    assert turns[0].transcript == "Hello I need an appointment"


@pytest.mark.asyncio
async def test_media_stream_invalid_session_closes(telephony_client):
    """WebSocket for non-existent session closes immediately with 1008."""
    from starlette.testclient import TestClient

    app = telephony_client
    client = TestClient(app, raise_server_exceptions=False)

    with pytest.raises(Exception):
        with client.websocket_connect("/ws/telephony/does-not-exist?token=bad-token") as ws:
            ws.receive_text()
