"""
Unit tests for voice_intake.demo_router.DemoRouter - intent classifier,
per-state dispatch, pending-patch contract, and proposal_id round-trip
invariant. Phase 4.5 regression coverage.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from voice_intake.db import (
    SQLAuditStore,
    create_tables,
    get_engine,
    make_session_factory,
)
from voice_intake.demo_router import (
    DemoRouter,
    Intent,
    _AppointmentPatch,
    _is_correction,
    _is_identity_refusal,
    _looks_like_question,
    classify_intent,
)
from voice_intake.models import (
    AppointmentRequest,
    AppointmentStatus,
    CallState,
    InsuranceNetworkStatus,
)
from voice_intake.policy import PolicyEngine, demo_policy_profile
from voice_intake.templates import DEFAULT_TEMPLATE_BUNDLE
from voice_intake.validator import ProposalValidator


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def demo_router_with_store(tmp_path):
    """Fresh DemoRouter wired to an isolated SQLite file (tmp_path).

    Using a file-backed sqlite (not :memory:) avoids the
    "no such table" cross-connection trap when the router opens its own
    SQLAuditStore from the session_factory.
    """
    db_path = tmp_path / "test.db"
    engine = get_engine(f"sqlite:///{db_path}")
    create_tables(engine)
    factory = make_session_factory(engine)
    router = DemoRouter(session_factory=factory, audit_secret="test-secret")
    return router


@pytest.fixture
def session_id() -> str:
    return "session-test-1"


def _prompt(state: CallState, transcript: str, sid: str) -> dict:
    """Build a minimal prompt dict matching what /turn passes to DemoRouter."""
    return {
        "session_id": sid,
        "current_state": state.value,
        "recent_turns": [{"speaker": "caller", "transcript": transcript}],
    }


# ── classify_intent + helpers ────────────────────────────────────────────────


def test_classify_intent_emergency():
    assert classify_intent("I have chest pain") == Intent.EMERGENCY
    assert classify_intent("I can't breathe") == Intent.EMERGENCY
    assert classify_intent("Bleeding") == Intent.EMERGENCY
    assert classify_intent("heart attack") == Intent.EMERGENCY


def test_classify_intent_human():
    assert classify_intent("I want to talk to a person") == Intent.HUMAN
    assert classify_intent("real person please") == Intent.HUMAN


def test_classify_intent_why():
    assert classify_intent("Why?") == Intent.WHY
    assert classify_intent("why do you need that") == Intent.WHY


def test_classify_intent_affirmation():
    assert classify_intent("Yes") == Intent.AFFIRMATION
    assert classify_intent("yeah that's fine") == Intent.AFFIRMATION


def test_classify_intent_denial():
    assert classify_intent("No") == Intent.DENIAL
    assert classify_intent("that's wrong") == Intent.DENIAL


def test_classify_intent_not_okay_is_denial():
    """REGRESSION for §1.8: 'not okay' must be DENIAL, not AFFIRMATION."""
    assert classify_intent("not okay") == Intent.DENIAL
    assert classify_intent("Not okay") == Intent.DENIAL


def test_classify_intent_dont_schedule_is_denial():
    """Pure denial phrasing without 'want' (which would classify as REFUSAL)."""
    assert classify_intent("don't schedule") == Intent.DENIAL
    assert classify_intent("do not book") == Intent.DENIAL


def test_classify_intent_dont_want_to_give_is_refusal():
    """GUARDRAIL for §1.8 ordering: 'I don't want to give that' must stay
    REFUSAL (refusal-before-denial), not DENIAL - identity routing depends
    on REFUSAL classification."""
    assert classify_intent("I don't want to give that") == Intent.REFUSAL


def test_classify_intent_substantive_appointment():
    assert classify_intent("I need to make an appointment") == Intent.APPOINTMENT_INTENT
    assert classify_intent("foot pain") == Intent.SUBSTANTIVE


def test_is_identity_refusal_tokens():
    """Single-word identity refusal tokens at name/DOB."""
    assert _is_identity_refusal("none", Intent.SUBSTANTIVE)
    assert _is_identity_refusal("no", Intent.DENIAL)
    assert _is_identity_refusal("n/a", Intent.SUBSTANTIVE)
    assert _is_identity_refusal("na", Intent.SUBSTANTIVE)
    # Refusal intent always wins
    assert _is_identity_refusal("I refuse", Intent.REFUSAL)
    # Substantive content does not trip it
    assert not _is_identity_refusal("John Smith", Intent.SUBSTANTIVE)


def test_is_correction_phrases():
    assert _is_correction("actually I meant something else")
    assert _is_correction("wait, that was wrong")
    assert _is_correction("change that")
    assert not _is_correction("foot pain")
    assert not _is_correction("yes")


def test_looks_like_question_interrogative_starters():
    assert _looks_like_question("Do you take walk-ins?")
    assert _looks_like_question("What about Medicare?")
    assert _looks_like_question("How does this work?")


def test_looks_like_question_does_not_fire_on_trailing_question_mark():
    """GUARDRAIL: bare '?' on a one-word answer must NOT be off-script."""
    assert not _looks_like_question("Aetna?")
    assert not _looks_like_question("Foot pain?")


# ── DemoRouter.propose: per-state dispatch ───────────────────────────────────


def test_propose_at_opening_appointment_intent_returns_reason_prompt(
    demo_router_with_store, session_id
):
    proposal = demo_router_with_store.propose(
        _prompt(CallState.OPENING, "I need to make an appointment", session_id),
        source_turn_id="turn-1",
        proposal_id="pid-1",
    )
    assert proposal.template_id == "appointment_reason_prompt"
    assert proposal.requested_transition == CallState.REASON_FOR_VISIT


def test_propose_at_opening_why_returns_appointment_reason_prompt(
    demo_router_with_store, session_id
):
    """REGRESSION for Blocker 4: must NOT return field_explanation_for_reason
    (template not allowed at OPENING - would validator-reject)."""
    proposal = demo_router_with_store.propose(
        _prompt(CallState.OPENING, "Why?", session_id),
        source_turn_id="turn-1",
        proposal_id="pid-1",
    )
    assert proposal.template_id == "appointment_reason_prompt"
    assert proposal.template_id != "field_explanation_for_reason"


def test_propose_at_opening_why_passes_validator(demo_router_with_store, session_id):
    """REGRESSION for Blocker 4 at the validator boundary."""
    from voice_intake.models import CallSession, CallMode, StirShakenAttestation

    proposal = demo_router_with_store.propose(
        _prompt(CallState.OPENING, "Why?", session_id),
        source_turn_id="turn-1",
        proposal_id="pid-1",
    )
    profile = demo_policy_profile()
    engine = PolicyEngine(profile)
    validator = ProposalValidator(DEFAULT_TEMPLATE_BUNDLE, engine)
    session = CallSession(
        session_id=session_id,
        telephony_call_id="call-1",
        operator_id="op-1",
        current_state=CallState.OPENING,
        mode=CallMode.AI_LED,
        policy_profile_id="demo",
        stir_shaken_attestation=StirShakenAttestation.UNKNOWN,
    )
    result = validator.validate(session, proposal)
    assert result.validator_result.accepted is True


def test_propose_at_reason_substantive_returns_name_prompt_with_patch(
    demo_router_with_store, session_id
):
    proposal = demo_router_with_store.propose(
        _prompt(CallState.REASON_FOR_VISIT, "Foot pain", session_id),
        source_turn_id="turn-1",
        proposal_id="pid-1",
    )
    assert proposal.template_id == "patient_name_prompt"
    assert proposal.requested_transition == CallState.IDENTITY_CAPTURE
    # Pending patch records reason_for_visit
    assert "pid-1" in demo_router_with_store._pending
    assert demo_router_with_store._pending["pid-1"].reason_for_visit == "Foot pain"


def test_propose_at_reason_generic_appointment_intent_re_asks(
    demo_router_with_store, session_id
):
    proposal = demo_router_with_store.propose(
        _prompt(CallState.REASON_FOR_VISIT, "I need an appointment", session_id),
        source_turn_id="turn-1",
        proposal_id="pid-1",
    )
    assert proposal.template_id == "appointment_reason_prompt"
    assert proposal.requested_transition == CallState.REASON_FOR_VISIT
    # No patch - generic intent isn't substantive
    assert "pid-1" not in demo_router_with_store._pending


def test_propose_at_identity_none_routes_to_human_takeover(
    demo_router_with_store, session_id
):
    """REGRESSION for Blocker 3: 'none' at IDENTITY → cannot_schedule_without_identity."""
    proposal = demo_router_with_store.propose(
        _prompt(CallState.IDENTITY_CAPTURE, "none", session_id),
        source_turn_id="turn-1",
        proposal_id="pid-1",
    )
    assert proposal.template_id == "cannot_schedule_without_identity"
    assert proposal.requested_transition == CallState.HUMAN_TAKEOVER


def test_propose_at_identity_no_routes_to_human_takeover(
    demo_router_with_store, session_id
):
    proposal = demo_router_with_store.propose(
        _prompt(CallState.IDENTITY_CAPTURE, "no", session_id),
        source_turn_id="turn-1",
        proposal_id="pid-1",
    )
    assert proposal.template_id == "cannot_schedule_without_identity"
    assert proposal.requested_transition == CallState.HUMAN_TAKEOVER


def test_propose_at_identity_dont_want_to_give_routes_to_handoff(
    demo_router_with_store, session_id
):
    """REGRESSION for §1.8 ordering: 'I don't want to give that' classified
    as REFUSAL (not DENIAL), so identity-refusal path fires."""
    proposal = demo_router_with_store.propose(
        _prompt(CallState.IDENTITY_CAPTURE, "I don't want to give that", session_id),
        source_turn_id="turn-1",
        proposal_id="pid-1",
    )
    assert proposal.template_id == "cannot_schedule_without_identity"
    assert proposal.requested_transition == CallState.HUMAN_TAKEOVER


def test_propose_at_demographics_none_routes_to_human_takeover(
    demo_router_with_store, session_id
):
    proposal = demo_router_with_store.propose(
        _prompt(CallState.DEMOGRAPHICS, "none", session_id),
        source_turn_id="turn-1",
        proposal_id="pid-1",
    )
    assert proposal.template_id == "cannot_schedule_without_identity"
    assert proposal.requested_transition == CallState.HUMAN_TAKEOVER


def test_propose_at_demographics_dob_parse_failure_clarifies(
    demo_router_with_store, session_id
):
    proposal = demo_router_with_store.propose(
        _prompt(CallState.DEMOGRAPHICS, "tomorrow", session_id),
        source_turn_id="turn-1",
        proposal_id="pid-1",
    )
    assert proposal.template_id == "dob_clarification"
    assert proposal.requested_transition == CallState.DEMOGRAPHICS
    assert "pid-1" not in demo_router_with_store._pending


def test_propose_at_demographics_invalid_calendar_clarifies(
    demo_router_with_store, session_id
):
    """REGRESSION for Medium 3.1: 2/31/01 must clarify, not store as DOB."""
    proposal = demo_router_with_store.propose(
        _prompt(CallState.DEMOGRAPHICS, "2/31/01", session_id),
        source_turn_id="turn-1",
        proposal_id="pid-1",
    )
    assert proposal.template_id == "dob_clarification"
    assert "pid-1" not in demo_router_with_store._pending


def test_propose_at_demographics_valid_dob_advances(
    demo_router_with_store, session_id
):
    proposal = demo_router_with_store.propose(
        _prompt(CallState.DEMOGRAPHICS, "8/7/1985", session_id),
        source_turn_id="turn-1",
        proposal_id="pid-1",
    )
    assert proposal.template_id == "insurance_prompt"
    assert proposal.requested_transition == CallState.INSURANCE
    assert demo_router_with_store._pending["pid-1"].date_of_birth == "1985-08-07"


def test_propose_at_insurance_aetna_returns_in_network(
    demo_router_with_store, session_id
):
    proposal = demo_router_with_store.propose(
        _prompt(CallState.INSURANCE, "Aetna", session_id),
        source_turn_id="turn-1",
        proposal_id="pid-1",
    )
    assert proposal.template_id == "insurance_in_network"
    assert proposal.variables["insurance_name"] == "Aetna"
    assert proposal.requested_transition == CallState.READBACK


def test_propose_at_insurance_kaiser_returns_out_of_network(
    demo_router_with_store, session_id
):
    proposal = demo_router_with_store.propose(
        _prompt(CallState.INSURANCE, "Kaiser", session_id),
        source_turn_id="turn-1",
        proposal_id="pid-1",
    )
    assert proposal.template_id == "insurance_out_of_network"
    assert proposal.variables["insurance_name"] == "Kaiser"


def test_propose_at_insurance_foocare_returns_unknown(
    demo_router_with_store, session_id
):
    proposal = demo_router_with_store.propose(
        _prompt(CallState.INSURANCE, "FooCare", session_id),
        source_turn_id="turn-1",
        proposal_id="pid-1",
    )
    assert proposal.template_id == "insurance_unknown"


def test_propose_at_insurance_none_routes_to_not_provided(
    demo_router_with_store, session_id
):
    """GUARDRAIL: 'none' at INSURANCE stays self-pay, NOT identity refusal."""
    proposal = demo_router_with_store.propose(
        _prompt(CallState.INSURANCE, "none", session_id),
        source_turn_id="turn-1",
        proposal_id="pid-1",
    )
    assert proposal.template_id == "insurance_not_provided"
    assert proposal.requested_transition == CallState.READBACK


def test_propose_at_insurance_why_still_explains_field(
    demo_router_with_store, session_id
):
    """THREE-WAY DISTINCTION (1 of 3) - WHY-intent precedes off-script-question detection."""
    proposal = demo_router_with_store.propose(
        _prompt(CallState.INSURANCE, "Why do you need that?", session_id),
        source_turn_id="turn-1",
        proposal_id="pid-1",
    )
    assert proposal.template_id == "field_explanation_for_insurance"
    assert proposal.requested_transition == CallState.INSURANCE


def test_propose_off_script_question_at_insurance_routes_to_handoff(
    demo_router_with_store, session_id
):
    """THREE-WAY DISTINCTION (2 of 3) - off-script question hands off."""
    proposal = demo_router_with_store.propose(
        _prompt(CallState.INSURANCE, "Do you take walk-ins?", session_id),
        source_turn_id="turn-1",
        proposal_id="pid-1",
    )
    assert proposal.template_id == "human_handoff"
    assert proposal.requested_transition == CallState.HUMAN_TAKEOVER


def test_propose_at_insurance_aetna_with_trailing_question_mark(
    demo_router_with_store, session_id
):
    """THREE-WAY DISTINCTION (3 of 3) - uncertain answer stays in network."""
    proposal = demo_router_with_store.propose(
        _prompt(CallState.INSURANCE, "Aetna?", session_id),
        source_turn_id="turn-1",
        proposal_id="pid-1",
    )
    assert proposal.template_id == "insurance_in_network"


def test_propose_at_insurance_deductible_phrase_still_checks_insurance(
    demo_router_with_store, session_id
):
    """Guardrail: billing words at INSURANCE must not become OOS handoff."""
    proposal = demo_router_with_store.propose(
        _prompt(CallState.INSURANCE, "Aetna PPO with $500 deductible", session_id),
        source_turn_id="turn-1",
        proposal_id="pid-1",
    )
    assert proposal.template_id == "insurance_unknown"
    assert proposal.requested_transition == CallState.READBACK


def test_propose_at_insurance_what_about_question_routes_to_handoff(
    demo_router_with_store, session_id
):
    proposal = demo_router_with_store.propose(
        _prompt(CallState.INSURANCE, "What about Medicare?", session_id),
        source_turn_id="turn-1",
        proposal_id="pid-1",
    )
    assert proposal.template_id == "human_handoff"


# ── READBACK ─────────────────────────────────────────────────────────────────


def _seed_appointment(router: DemoRouter, sid: str, status: InsuranceNetworkStatus,
                      name: str = "Aetna") -> None:
    """Pre-populate an appointment row simulating that we've reached READBACK."""
    appt = AppointmentRequest(
        appointment_id="appt-test",
        session_id=sid,
        reason_for_visit="foot pain",
        patient_name="John Smith",
        date_of_birth="1985-08-07",
        insurance_name=name,
        insurance_network_status=status,
        appointment_status=AppointmentStatus.COLLECTING,
    )
    router._store.upsert_appointment_request(appt)


def test_propose_at_readback_yes_in_network_schedules(
    demo_router_with_store, session_id
):
    _seed_appointment(demo_router_with_store, session_id, InsuranceNetworkStatus.IN_NETWORK)
    proposal = demo_router_with_store.propose(
        _prompt(CallState.READBACK, "yes", session_id),
        source_turn_id="turn-1",
        proposal_id="pid-1",
    )
    assert proposal.template_id == "appointment_scheduled"
    patch = demo_router_with_store._pending["pid-1"]
    assert patch.appointment_status == AppointmentStatus.SCHEDULED
    assert patch.scheduled_slot == "next_available_demo_slot"


def test_propose_at_readback_yes_unknown_creates_request(
    demo_router_with_store, session_id
):
    _seed_appointment(demo_router_with_store, session_id, InsuranceNetworkStatus.UNKNOWN, name="FooCare")
    proposal = demo_router_with_store.propose(
        _prompt(CallState.READBACK, "yes", session_id),
        source_turn_id="turn-1",
        proposal_id="pid-1",
    )
    assert proposal.template_id == "appointment_request_created"
    patch = demo_router_with_store._pending["pid-1"]
    assert patch.appointment_status == AppointmentStatus.STAFF_REVIEW


def test_propose_at_readback_no_routes_to_human_takeover(
    demo_router_with_store, session_id
):
    _seed_appointment(demo_router_with_store, session_id, InsuranceNetworkStatus.IN_NETWORK)
    proposal = demo_router_with_store.propose(
        _prompt(CallState.READBACK, "no", session_id),
        source_turn_id="turn-1",
        proposal_id="pid-1",
    )
    assert proposal.template_id == "human_handoff"
    assert proposal.requested_transition == CallState.HUMAN_TAKEOVER


def test_propose_at_readback_unknown_unclear_response_re_asks_unknown(
    demo_router_with_store, session_id
):
    """REGRESSION for Blocker 5: UNKNOWN insurance + unclear → insurance_unknown."""
    _seed_appointment(demo_router_with_store, session_id, InsuranceNetworkStatus.UNKNOWN, name="FooCare")
    proposal = demo_router_with_store.propose(
        _prompt(CallState.READBACK, "wait what?", session_id),
        source_turn_id="turn-1",
        proposal_id="pid-1",
    )
    # "wait" is correction language; correction at READBACK → handoff. But since
    # the network status is UNKNOWN and the unclear-response fallback handles
    # everything BEFORE correction, behavior depends on dispatch order. Per the
    # plan, correction at READBACK → handoff (§1.7a is correct).
    # However, the more useful regression here is the non-correction unclear
    # case - use a phrase that doesn't trigger correction.
    # Re-test with non-correction unclear input:
    proposal = demo_router_with_store.propose(
        _prompt(CallState.READBACK, "could you repeat", session_id),
        source_turn_id="turn-2",
        proposal_id="pid-2",
    )
    assert proposal.template_id == "insurance_unknown"
    assert proposal.requested_transition == CallState.READBACK


def test_propose_at_readback_not_okay_does_not_schedule(
    demo_router_with_store, session_id
):
    """REGRESSION for §1.8: 'not okay' at READBACK → handoff, NOT schedule."""
    _seed_appointment(demo_router_with_store, session_id, InsuranceNetworkStatus.IN_NETWORK)
    proposal = demo_router_with_store.propose(
        _prompt(CallState.READBACK, "not okay", session_id),
        source_turn_id="turn-1",
        proposal_id="pid-1",
    )
    assert proposal.template_id == "human_handoff"
    assert proposal.template_id != "appointment_scheduled"


def test_propose_at_readback_dont_schedule_does_not_schedule(
    demo_router_with_store, session_id
):
    _seed_appointment(demo_router_with_store, session_id, InsuranceNetworkStatus.IN_NETWORK)
    proposal = demo_router_with_store.propose(
        _prompt(CallState.READBACK, "I don't want to schedule", session_id),
        source_turn_id="turn-1",
        proposal_id="pid-1",
    )
    assert proposal.template_id == "human_handoff"


# ── Emergency / Human / Correction at any active state ──────────────────────


@pytest.mark.parametrize("state", [
    CallState.REASON_FOR_VISIT,
    CallState.IDENTITY_CAPTURE,
    CallState.DEMOGRAPHICS,
    CallState.INSURANCE,
    CallState.READBACK,
])
def test_propose_emergency_from_any_state(demo_router_with_store, session_id, state):
    proposal = demo_router_with_store.propose(
        _prompt(state, "I have chest pain", session_id),
        source_turn_id="turn-1",
        proposal_id="pid-1",
    )
    assert proposal.template_id == "emergency_redirect"
    assert proposal.requested_transition == CallState.EMERGENCY_EXIT


@pytest.mark.parametrize("state", [
    CallState.REASON_FOR_VISIT,
    CallState.IDENTITY_CAPTURE,
    CallState.DEMOGRAPHICS,
    CallState.INSURANCE,
    CallState.READBACK,
])
def test_propose_human_handoff_from_any_state(demo_router_with_store, session_id, state):
    proposal = demo_router_with_store.propose(
        _prompt(state, "I want to talk to a person", session_id),
        source_turn_id="turn-1",
        proposal_id="pid-1",
    )
    assert proposal.template_id == "human_handoff"
    assert proposal.requested_transition == CallState.HUMAN_TAKEOVER


@pytest.mark.parametrize("state", [
    CallState.IDENTITY_CAPTURE,
    CallState.DEMOGRAPHICS,
    CallState.INSURANCE,
    CallState.READBACK,
])
def test_propose_correction_at_field_states_routes_to_handoff(
    demo_router_with_store, session_id, state
):
    """REGRESSION for §1.7a: correction at field/readback states hands off."""
    proposal = demo_router_with_store.propose(
        _prompt(state, "actually that was wrong", session_id),
        source_turn_id="turn-1",
        proposal_id="pid-1",
    )
    assert proposal.template_id == "human_handoff"
    assert proposal.requested_transition == CallState.HUMAN_TAKEOVER


def test_propose_correction_at_opening_does_not_handoff(
    demo_router_with_store, session_id
):
    """GUARDRAIL for §1.7a: correction at OPENING is not escalated."""
    proposal = demo_router_with_store.propose(
        _prompt(CallState.OPENING, "actually I need an appointment", session_id),
        source_turn_id="turn-1",
        proposal_id="pid-1",
    )
    assert proposal.template_id == "appointment_reason_prompt"


def test_propose_correction_at_reason_does_not_handoff(
    demo_router_with_store, session_id
):
    """GUARDRAIL for §1.7a: correction at REASON_FOR_VISIT is not escalated.
    Words like 'change' / 'actually' can innocuously appear inside a substantive
    medical reason."""
    proposal = demo_router_with_store.propose(
        _prompt(CallState.REASON_FOR_VISIT, "change in vision", session_id),
        source_turn_id="turn-1",
        proposal_id="pid-1",
    )
    assert proposal.template_id == "patient_name_prompt"
    assert proposal.requested_transition == CallState.IDENTITY_CAPTURE


# ── Pending-patch + proposal_id invariants ──────────────────────────────────


def test_proposal_id_round_trip(demo_router_with_store, session_id):
    """The proposal_id passed in must equal ModelProposal.proposal_id AND
    be the key in _pending."""
    pid = "round-trip-id-xyz"
    proposal = demo_router_with_store.propose(
        _prompt(CallState.REASON_FOR_VISIT, "Foot pain", session_id),
        source_turn_id="turn-1",
        proposal_id=pid,
    )
    assert proposal.proposal_id == pid
    assert pid in demo_router_with_store._pending


def test_commit_applies_pending_patch_and_removes_from_dict(
    demo_router_with_store, session_id
):
    pid = "pid-commit"
    demo_router_with_store.propose(
        _prompt(CallState.REASON_FOR_VISIT, "Foot pain", session_id),
        source_turn_id="turn-1",
        proposal_id=pid,
    )
    assert pid in demo_router_with_store._pending

    demo_router_with_store.commit(pid)

    assert pid not in demo_router_with_store._pending
    appt = demo_router_with_store._store.get_appointment_request(session_id)
    assert appt is not None
    assert appt.reason_for_visit == "Foot pain"


def test_discard_drops_patch_without_writing(demo_router_with_store, session_id):
    pid = "pid-discard"
    demo_router_with_store.propose(
        _prompt(CallState.REASON_FOR_VISIT, "Foot pain", session_id),
        source_turn_id="turn-1",
        proposal_id=pid,
    )
    assert pid in demo_router_with_store._pending

    demo_router_with_store.discard(pid)

    assert pid not in demo_router_with_store._pending
    # No row should have been created
    appt = demo_router_with_store._store.get_appointment_request(session_id)
    assert appt is None


def test_prune_removes_stale_patches(demo_router_with_store, session_id):
    """Patches older than PATCH_TTL_SECONDS get cleaned on next propose()."""
    # Hand-construct a stale patch
    stale_pid = "stale-pid"
    stale_patch = _AppointmentPatch(
        session_id=session_id,
        created_at=datetime.now(timezone.utc) - timedelta(hours=2),
        reason_for_visit="old",
    )
    demo_router_with_store._pending[stale_pid] = stale_patch

    # Add a fresh patch and propose() - the prune runs at the top of propose
    fresh_pid = "fresh-pid"
    demo_router_with_store.propose(
        _prompt(CallState.REASON_FOR_VISIT, "Foot pain", session_id),
        source_turn_id="turn-1",
        proposal_id=fresh_pid,
    )
    assert stale_pid not in demo_router_with_store._pending
    assert fresh_pid in demo_router_with_store._pending


def test_propose_through_validator_no_rejection_on_happy_path(
    demo_router_with_store, session_id
):
    """For each demo state, the proposal DemoRouter emits passes ProposalValidator."""
    from voice_intake.models import CallSession, CallMode, StirShakenAttestation

    profile = demo_policy_profile()
    engine = PolicyEngine(profile)
    validator = ProposalValidator(DEFAULT_TEMPLATE_BUNDLE, engine)

    flows = [
        (CallState.OPENING, "I need an appointment"),
        (CallState.REASON_FOR_VISIT, "Foot pain"),
        (CallState.IDENTITY_CAPTURE, "John Smith"),
        (CallState.DEMOGRAPHICS, "8/7/1985"),
        (CallState.INSURANCE, "Aetna"),
    ]

    for idx, (state, transcript) in enumerate(flows):
        # Each step gets its own fresh session in the right state.
        session = CallSession(
            session_id=f"{session_id}-{idx}",
            telephony_call_id="call-1",
            operator_id="op-1",
            current_state=state,
            mode=CallMode.AI_LED,
            policy_profile_id="demo",
            stir_shaken_attestation=StirShakenAttestation.UNKNOWN,
        )
        proposal = demo_router_with_store.propose(
            _prompt(state, transcript, session.session_id),
            source_turn_id=f"turn-{idx}",
            proposal_id=f"pid-{idx}",
        )
        result = validator.validate(session, proposal)
        assert result.validator_result.accepted, (
            f"Rejected at {state.value}: template={proposal.template_id}, "
            f"transition={proposal.requested_transition.value}"
        )
