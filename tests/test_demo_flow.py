"""
End-to-end tests for the demo scheduling flow via FastAPI TestClient with
DEMO_ROUTER=true. Phase 4.5 regression coverage at the route boundary.

Test-mocking rule: normal-flow tests use the real app-state DemoRouter.
Only test_rejected_demo_router_proposal_does_not_mutate_appointment
monkeypatches DemoRouter.propose, and even there it monkeypatches the SAME
app.state.demo_router instance the route uses.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from voice_intake.api.app import create_app
from voice_intake.api.deps import get_store
from voice_intake.config import Settings
from voice_intake.db import SQLAuditStore, create_tables, get_engine, make_session_factory
from voice_intake.demo_router import DemoRouter
from voice_intake.models import CallState, ModelProposal


def _mock_embed(texts: list[str]) -> list[list[float]]:
    return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


@pytest.fixture
def demo_app(tmp_path):
    """FastAPI app with DEMO_ROUTER=true wired to a file-backed SQLite.

    File-backed SQLite (not :memory:) avoids the
    "no such table in another connection" trap when DemoRouter opens its own
    SQLAuditStore from the shared session_factory.
    """
    db_path = tmp_path / "demo.db"
    test_settings = Settings(
        database_url=f"sqlite:///{db_path}",
        demo_router=True,
        mock_llm=True,
        mock_asr=True,
        mock_rag=True,
        audit_hmac_secret="test-audit-secret",
        stream_auth_secret="test-stream-secret",
    )
    app = create_app(settings=test_settings)
    app.state.embed_fn = _mock_embed

    # Build a single shared engine + factory + store. Override get_store
    # so every request uses the same store. Also wire app.state.session_factory
    # so DemoRouter (constructed below) uses the same engine.
    engine = get_engine(f"sqlite:///{db_path}")
    create_tables(engine)
    factory = make_session_factory(engine)
    store = SQLAuditStore(factory, hmac_secret="test-audit-secret")

    app.state.session_factory = factory
    app.dependency_overrides[get_store] = lambda: store
    # Manually wire app.state.demo_router because the app's lifespan didn't
    # run before TestClient (test fixture creates the app outside lifespan).
    app.state.demo_router = DemoRouter(
        session_factory=factory, audit_secret="test-audit-secret"
    )
    return app, store


@pytest.fixture
def client(demo_app):
    app, _ = demo_app
    return TestClient(app, raise_server_exceptions=True)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _open_session(client: TestClient) -> str:
    resp = client.post("/session/open", json={
        "telephony_call_id": "demo-call-1",
        "operator_id": "demo",
        "stir_shaken_attestation": "unknown",
    })
    assert resp.status_code == 201, resp.text
    return resp.json()["session_id"]


def _turn(client: TestClient, sid: str, transcript: str) -> dict:
    resp = client.post(f"/session/{sid}/turn", json={"transcript": transcript})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _appt(client: TestClient, sid: str) -> tuple[int, dict | None]:
    resp = client.get(f"/session/{sid}/appointment")
    if resp.status_code == 404:
        return 404, None
    assert resp.status_code == 200, resp.text
    return 200, resp.json()


# ── Opening / entry state ───────────────────────────────────────────────────


def test_demo_session_opens_at_opening_state(client):
    """REGRESSION: with DEMO_ROUTER=true, /session/open enters at OPENING."""
    sid = _open_session(client)
    resp = client.get(f"/session/{sid}")
    assert resp.json()["current_state"] == CallState.OPENING.value


def test_appointment_404_before_first_commit_then_200_after(client):
    sid = _open_session(client)
    status, _ = _appt(client, sid)
    assert status == 404

    # First substantive turn at OPENING moves to REASON_FOR_VISIT but doesn't
    # commit a patch yet (no field collected). A reason-substantive turn DOES
    # commit a patch.
    _turn(client, sid, "I need an appointment")
    status, _ = _appt(client, sid)
    assert status == 404  # OPENING transition has no patch

    _turn(client, sid, "Foot pain")  # at REASON_FOR_VISIT, commits patch
    status, body = _appt(client, sid)
    assert status == 200
    assert body["reason_for_visit"] == "Foot pain"


# ── Frozen acceptance-criteria flows ─────────────────────────────────────────


def test_aetna_full_flow_schedules_appointment(client):
    sid = _open_session(client)
    _turn(client, sid, "I need to make an appointment")
    _turn(client, sid, "Foot pain")
    _turn(client, sid, "John Smith")
    _turn(client, sid, "8/7/01")
    _turn(client, sid, "Aetna")
    final = _turn(client, sid, "Yes")
    assert final["voice_action"]["template_id"] == "appointment_scheduled"
    status, body = _appt(client, sid)
    assert status == 200
    assert body["appointment_status"] == "scheduled"
    assert body["scheduled_slot"] == "next_available_demo_slot"
    assert body["insurance_name"] == "Aetna"
    assert body["insurance_network_status"] == "in_network"
    assert body["reason_for_visit"] == "Foot pain"


def test_kaiser_full_flow_oon_warning_then_schedules(client):
    sid = _open_session(client)
    _turn(client, sid, "I need an appointment")
    _turn(client, sid, "Back pain")
    _turn(client, sid, "Jane Doe")
    _turn(client, sid, "1/1/1990")
    response = _turn(client, sid, "Kaiser")
    assert response["voice_action"]["template_id"] == "insurance_out_of_network"
    final = _turn(client, sid, "Yes")
    assert final["voice_action"]["template_id"] == "appointment_scheduled"
    _, body = _appt(client, sid)
    assert body["insurance_network_status"] == "out_of_network"
    assert body["appointment_status"] == "scheduled"


def test_foocare_full_flow_creates_staff_review(client):
    sid = _open_session(client)
    _turn(client, sid, "I need an appointment")
    _turn(client, sid, "Headache")
    _turn(client, sid, "Bob Lee")
    _turn(client, sid, "5/1/1980")
    response = _turn(client, sid, "FooCare")
    assert response["voice_action"]["template_id"] == "insurance_unknown"
    final = _turn(client, sid, "Yes")
    # CRITICAL: must NOT emit appointment_scheduled - the panel says staff_review,
    # so the spoken text must agree.
    assert final["voice_action"]["template_id"] == "appointment_request_created"
    _, body = _appt(client, sid)
    assert body["appointment_status"] == "staff_review"
    assert body["insurance_network_status"] == "unknown"


def test_self_pay_full_flow_creates_staff_review_via_insurance(client):
    """REGRESSION for Blocker 3: 'none' at INSURANCE → not_provided → staff_review."""
    sid = _open_session(client)
    _turn(client, sid, "I need an appointment")
    _turn(client, sid, "Sore throat")
    _turn(client, sid, "Carla Park")
    _turn(client, sid, "3/3/1995")
    response = _turn(client, sid, "none")
    assert response["voice_action"]["template_id"] == "insurance_not_provided"
    final = _turn(client, sid, "Yes")
    assert final["voice_action"]["template_id"] == "appointment_request_created"
    _, body = _appt(client, sid)
    assert body["appointment_status"] == "staff_review"
    assert body["insurance_network_status"] == "not_provided"


# ── Interruption regressions ────────────────────────────────────────────────


def test_emergency_at_demographics_redirects(client):
    sid = _open_session(client)
    _turn(client, sid, "appointment please")
    _turn(client, sid, "Foot pain")
    _turn(client, sid, "John Smith")
    response = _turn(client, sid, "I have chest pain")
    assert response["voice_action"]["template_id"] == "emergency_redirect"
    assert response["current_state"] == CallState.EMERGENCY_EXIT.value


def test_safety_prerouter_bare_bleeding_redirects_at_opening(client):
    sid = _open_session(client)
    response = _turn(client, sid, "Bleeding")
    assert response["voice_action"]["template_id"] == "emergency_redirect"
    assert response["current_state"] == CallState.EMERGENCY_EXIT.value


def test_safety_prerouter_currently_bleeding_redirects_at_reason(client):
    sid = _open_session(client)
    _turn(client, sid, "I need an appointment")
    response = _turn(client, sid, "Currently bleeding")
    assert response["voice_action"]["template_id"] == "emergency_redirect"
    assert response["current_state"] == CallState.EMERGENCY_EXIT.value


def test_safety_prerouter_balance_handoff_at_opening(client):
    sid = _open_session(client)
    response = _turn(client, sid, "I want to check my balance")
    assert response["voice_action"]["template_id"] == "human_handoff"
    assert response["current_state"] == CallState.HUMAN_TAKEOVER.value


def test_safety_prerouter_balance_handoff_at_reason(client):
    sid = _open_session(client)
    _turn(client, sid, "I need an appointment")
    response = _turn(client, sid, "No visit, just checking balance")
    assert response["voice_action"]["template_id"] == "human_handoff"
    assert response["current_state"] == CallState.HUMAN_TAKEOVER.value


def test_human_handoff_at_insurance_alerts_staff(client):
    sid = _open_session(client)
    _turn(client, sid, "I need an appointment")
    _turn(client, sid, "Foot pain")
    _turn(client, sid, "John Smith")
    _turn(client, sid, "1/1/1990")
    response = _turn(client, sid, "Can I talk to a person?")
    assert response["voice_action"]["template_id"] == "human_handoff"
    assert response["current_state"] == CallState.HUMAN_TAKEOVER.value


def test_aetna_deductible_phrase_at_insurance_does_not_oos_handoff(client):
    sid = _open_session(client)
    _turn(client, sid, "I need an appointment")
    _turn(client, sid, "Foot pain")
    _turn(client, sid, "John Smith")
    _turn(client, sid, "1/1/1990")
    response = _turn(client, sid, "Aetna PPO with $500 deductible")
    assert response["voice_action"]["template_id"] == "insurance_unknown"
    assert response["current_state"] == CallState.READBACK.value


def test_why_at_each_field_state_returns_field_specific_explanation(client):
    """All four field-state 'why?' responses use the right field-specific template."""
    expected = {
        CallState.REASON_FOR_VISIT.value: "field_explanation_for_reason",
        CallState.IDENTITY_CAPTURE.value: "field_explanation_for_name",
        CallState.DEMOGRAPHICS.value: "field_explanation_for_dob",
        CallState.INSURANCE.value: "field_explanation_for_insurance",
    }

    for state, expected_template in expected.items():
        sid = _open_session(client)
        # Walk to the target state with substantive answers.
        _turn(client, sid, "I need an appointment")  # → REASON_FOR_VISIT
        if state != CallState.REASON_FOR_VISIT.value:
            _turn(client, sid, "Foot pain")           # → IDENTITY_CAPTURE
        if state in (CallState.DEMOGRAPHICS.value, CallState.INSURANCE.value):
            _turn(client, sid, "John Smith")          # → DEMOGRAPHICS
        if state == CallState.INSURANCE.value:
            _turn(client, sid, "1/1/1990")            # → INSURANCE

        response = _turn(client, sid, "Why?")
        assert response["voice_action"]["template_id"] == expected_template


def test_why_at_opening_passes_validator(client):
    """REGRESSION for Blocker 4 at the API boundary: 'why?' at OPENING returns
    a valid voice_action, NOT a validator rejection."""
    sid = _open_session(client)
    response = _turn(client, sid, "Why?")
    assert response["voice_action"] is not None
    assert response["rejection"] is None
    assert response["voice_action"]["template_id"] == "appointment_reason_prompt"


@pytest.mark.parametrize("state, walk", [
    (CallState.IDENTITY_CAPTURE, [("I need an appointment", None), ("Foot pain", None)]),
    (CallState.DEMOGRAPHICS, [("appointment please", None), ("Foot pain", None), ("John Smith", None)]),
    (CallState.INSURANCE, [("appointment please", None), ("Foot pain", None), ("John Smith", None), ("1/1/1990", None)]),
])
def test_correction_at_field_states_routes_to_handoff(client, state, walk):
    """REGRESSION for §1.7a: correction at IDENTITY/DEMOGRAPHICS/INSURANCE/READBACK → handoff."""
    sid = _open_session(client)
    for transcript, _ in walk:
        _turn(client, sid, transcript)
    response = _turn(client, sid, "actually that was wrong")
    assert response["voice_action"]["template_id"] == "human_handoff"
    assert response["current_state"] == CallState.HUMAN_TAKEOVER.value


def test_off_script_question_at_insurance_does_not_become_unknown_insurance(client):
    """REGRESSION for §1.7b: 'Do you take walk-ins?' at INSURANCE → handoff;
    appointment.insurance_name is NOT set to the question text."""
    sid = _open_session(client)
    _turn(client, sid, "I need an appointment")
    _turn(client, sid, "Foot pain")
    _turn(client, sid, "John Smith")
    _turn(client, sid, "1/1/1990")
    response = _turn(client, sid, "Do you take walk-ins?")
    assert response["voice_action"]["template_id"] == "human_handoff"

    # Appointment must not have stored "Do You Take Walk-ins" as insurance_name.
    _, body = _appt(client, sid)
    assert body["insurance_name"] is None or "walk" not in (body["insurance_name"] or "").lower()


def test_readback_not_okay_does_not_schedule(client):
    """REGRESSION for §1.8: 'not okay' at Aetna readback → STAFF_REVIEW + handoff,
    NOT appointment_scheduled."""
    sid = _open_session(client)
    _turn(client, sid, "I need an appointment")
    _turn(client, sid, "Foot pain")
    _turn(client, sid, "John Smith")
    _turn(client, sid, "1/1/1990")
    _turn(client, sid, "Aetna")
    final = _turn(client, sid, "not okay")
    assert final["voice_action"]["template_id"] == "human_handoff"
    assert final["voice_action"]["template_id"] != "appointment_scheduled"


def test_invalid_dob_loops_with_clarification(client):
    """REGRESSION for Medium 3.1: invalid calendar date → dob_clarification, not stored."""
    sid = _open_session(client)
    _turn(client, sid, "I need an appointment")
    _turn(client, sid, "Foot pain")
    _turn(client, sid, "John Smith")
    response = _turn(client, sid, "2/31/01")
    assert response["voice_action"]["template_id"] == "dob_clarification"
    assert response["current_state"] == CallState.DEMOGRAPHICS.value


# ── Architecture-boundary regression ────────────────────────────────────────


def test_rejected_demo_router_proposal_does_not_mutate_appointment(demo_app):
    """ARCHITECTURE BOUNDARY: a proposal that gets validator-rejected must NOT
    leave any AppointmentRequest row behind. Proves the no-mutation-before-
    validation invariant at the public-API boundary, not just at unit level.

    Setup must monkeypatch the SAME app.state.demo_router instance that
    /turn calls discard() on - otherwise the test gives false confidence.
    """
    app, store = demo_app
    client = TestClient(app, raise_server_exceptions=True)
    sid = _open_session(client)

    demo_router = app.state.demo_router  # SAME singleton as /turn uses
    real_propose = demo_router.propose

    def _propose_invalid(prompt, source_turn_id, proposal_id=None):
        # Use the real implementation to record a pending patch (so we can
        # observe whether discard() actually fires), but override the
        # template_id to one that's not allowed at OPENING → guaranteed reject.
        proposal = real_propose(prompt, source_turn_id, proposal_id=proposal_id)
        return ModelProposal(
            proposal_id=proposal.proposal_id,
            source_turn_id=proposal.source_turn_id,
            template_id="readback_confirmation",  # not allowed at OPENING
            variables={},  # also missing required vars → another reject
            requested_transition=CallState.READBACK,
            model_version=proposal.model_version,
            session_id=proposal.session_id,
        )

    demo_router.propose = _propose_invalid
    try:
        resp = client.post(f"/session/{sid}/turn", json={"transcript": "I need an appointment"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # Validator must reject
        assert body["rejection"] is not None
        assert body["voice_action"] is None

        # No appointment row may exist
        appt_resp = client.get(f"/session/{sid}/appointment")
        assert appt_resp.status_code == 404

        # Pending patches must have been discarded - no stale entries
        assert len(demo_router._pending) == 0, (
            f"Pending patch leaked: {list(demo_router._pending.keys())}"
        )
    finally:
        demo_router.propose = real_propose


# ── Banned-string scope assertion ───────────────────────────────────────────


_BANNED_STRINGS = [
    "consent",
    "address",
    "phone number",
    "social security",
    "ssn",
    "member id",
    "recorded for staff follow-up",
    "****1985",
]


def test_no_banned_strings_in_demo_proposals_or_appointment_data(client):
    """Banned-string scope = runtime output (rendered template content +
    appointment-row fields). Does NOT scan the entire repo - templates.py
    legitimately retains consent templates for the non-demo path."""
    from voice_intake.templates import DEFAULT_TEMPLATE_BUNDLE

    sid = _open_session(client)
    _turn(client, sid, "I need an appointment")
    _turn(client, sid, "Foot pain")
    _turn(client, sid, "John Smith")
    _turn(client, sid, "8/7/01")
    aetna_response = _turn(client, sid, "Aetna")
    final = _turn(client, sid, "Yes")

    # Render every emitted template with its variables and check no banned
    # string appears in the rendered AI text.
    rendered_texts: list[str] = []
    for resp in (aetna_response, final):
        action = resp["voice_action"]
        if action is None:
            continue
        template = DEFAULT_TEMPLATE_BUNDLE.template_for(action["template_id"])
        assert template is not None
        text = template.content
        for k, v in action["allowed_variables"].items():
            text = text.replace(f"{{{{{k}}}}}", v)
        rendered_texts.append(text.lower())

    # Also include the appointment-row data
    _, body = _appt(client, sid)
    for value in body.values():
        if isinstance(value, str):
            rendered_texts.append(value.lower())

    blob = " ".join(rendered_texts)
    for banned in _BANNED_STRINGS:
        assert banned.lower() not in blob, f"Banned string {banned!r} found in demo output"
