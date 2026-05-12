from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from voice_intake.api.app import create_app
from voice_intake.api.deps import get_store
from voice_intake.boundary import (
    PromptBoundaryError,
    inspect_prompt_payload,
    load_prompt_boundary_policy,
    sanitize_prompt_payload,
)
from voice_intake.config import Settings
from voice_intake.db import SQLAuditStore, create_tables, get_engine, make_session_factory
from voice_intake.demo_router import DemoRouter
from voice_intake.models import (
    CallSession,
    CallState,
    FieldCandidate,
    ModelProposal,
    Speaker,
    VerificationStatus,
    VoiceTurn,
)
from voice_intake.prompting import build_model_prompt
from voice_intake.templates import DEFAULT_TEMPLATE_BUNDLE


def _candidate(field_name: str = "member_id", value: str = "123456789") -> FieldCandidate:
    return FieldCandidate(
        field_name=field_name,
        candidate_value=value,
        confidence=0.99,
        verification_status=VerificationStatus.PENDING_CONFIRMATION,
        source_turn_ids=["t1"],
        sensitivity_class="phi",
    )


def _turn(transcript: str) -> VoiceTurn:
    ts = datetime.now(timezone.utc)
    return VoiceTurn(
        turn_id="t1",
        speaker=Speaker.CALLER,
        audio_ref=None,
        retention_policy_id="default",
        transcript=transcript,
        partial_or_final="final",
        asr_confidence=0.98,
        barge_in=False,
        start_ts=ts,
        end_ts=ts,
    )


def _policy():
    return load_prompt_boundary_policy(Path("config/prompt_boundary_policy.yaml"))


def test_prompt_boundary_policy_loads_config_and_hashes_source():
    policy_path = Path("config/prompt_boundary_policy.yaml")
    policy = load_prompt_boundary_policy(policy_path)

    assert policy.source_path == policy_path
    assert len(policy.source_hash) == 64
    assert "member_id" in policy.forbidden_field_types
    assert policy.forbidden_patterns


def test_sanitize_prompt_payload_redacts_field_candidates():
    prompt = {
        "recent_turns": [
            {"speaker": "caller", "transcript": "My member ID is 123456789"}
        ]
    }

    sanitized = sanitize_prompt_payload(prompt, [_candidate()], _policy())

    serialized = json.dumps(sanitized)
    assert "123456789" not in serialized
    assert "[MEMBER_ID]" in serialized
    assert prompt["recent_turns"][0]["transcript"] == "My member ID is 123456789"


def test_sanitize_prompt_payload_redacts_forbidden_patterns():
    prompt = {
        "recent_turns": [
            {"speaker": "caller", "transcript": "The number is 987654321"}
        ]
    }

    sanitized = sanitize_prompt_payload(prompt, [], _policy())

    serialized = json.dumps(sanitized)
    assert "987654321" not in serialized
    assert "[REDACTED]" in serialized


def test_inspect_prompt_payload_rejects_raw_candidate_values():
    prompt = {
        "recent_turns": [
            {"speaker": "caller", "transcript": "My member ID is 123456789"}
        ]
    }

    try:
        inspect_prompt_payload(prompt, [_candidate()], _policy())
    except PromptBoundaryError as exc:
        assert "unmasked member_id" in str(exc)
    else:
        raise AssertionError("expected PromptBoundaryError")


def test_inspect_prompt_payload_rejects_forbidden_patterns():
    prompt = {
        "recent_turns": [
            {"speaker": "caller", "transcript": "The number is 987654321"}
        ]
    }

    try:
        inspect_prompt_payload(prompt, [], _policy())
    except PromptBoundaryError as exc:
        assert "forbidden pattern leaked" in str(exc)
    else:
        raise AssertionError("expected PromptBoundaryError")


def test_build_model_prompt_boundary_policy_prevents_long_digit_leak():
    session = CallSession(session_id="s1", telephony_call_id="c1", operator_id="o1")
    candidate = _candidate()

    prompt = build_model_prompt(
        session=session,
        recent_turns=[_turn("My member ID is 123456789")],
        field_candidates=[candidate],
        available_templates=[DEFAULT_TEMPLATE_BUNDLE.templates["collect_field_prompt"]],
        boundary_policy=_policy(),
    )

    serialized = json.dumps(prompt)
    assert "123456789" not in serialized
    assert "[MEMBER_ID]" in serialized


def test_existing_rejected_demo_router_proposal_integrity_still_passes(tmp_path):
    db_path = tmp_path / "demo.db"
    settings = Settings(
        database_url=f"sqlite:///{db_path}",
        demo_router=True,
        mock_llm=True,
        mock_asr=True,
        mock_rag=True,
        audit_hmac_secret="test-audit-secret",
        stream_auth_secret="test-stream-secret",
    )
    app = create_app(settings=settings)
    engine = get_engine(f"sqlite:///{db_path}")
    create_tables(engine)
    factory = make_session_factory(engine)
    store = SQLAuditStore(factory, hmac_secret="test-audit-secret")
    app.state.session_factory = factory
    app.dependency_overrides[get_store] = lambda: store
    app.state.demo_router = DemoRouter(
        session_factory=factory, audit_secret="test-audit-secret"
    )
    client = TestClient(app, raise_server_exceptions=True)
    open_response = client.post(
        "/session/open",
        json={
            "telephony_call_id": "demo-call-1",
            "operator_id": "demo",
            "stir_shaken_attestation": "unknown",
        },
    )
    session_id = open_response.json()["session_id"]

    demo_router = app.state.demo_router
    real_propose = demo_router.propose

    def _propose_invalid(prompt, source_turn_id, proposal_id=None):
        proposal = real_propose(prompt, source_turn_id, proposal_id=proposal_id)
        return ModelProposal(
            proposal_id=proposal.proposal_id,
            source_turn_id=proposal.source_turn_id,
            template_id="readback_confirmation",
            variables={},
            requested_transition=CallState.READBACK,
            model_version=proposal.model_version,
            session_id=proposal.session_id,
        )

    demo_router.propose = _propose_invalid
    try:
        response = client.post(
            f"/session/{session_id}/turn",
            json={"transcript": "I need an appointment"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["rejection"] is not None
        assert body["voice_action"] is None
        assert client.get(f"/session/{session_id}/appointment").status_code == 404
        assert len(demo_router._pending) == 0
    finally:
        demo_router.propose = real_propose
