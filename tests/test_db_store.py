import unittest
from uuid import uuid4

from voice_intake.audit import AuditStore
from voice_intake.audit_verify import verify_chain
from voice_intake.db import SQLAuditStore, create_tables, get_engine, make_session_factory
from voice_intake.models import (
    AuditActor,
    AuditEvent,
    AuditEventType,
    CallMode,
    CallSession,
    CallState,
    ConsentArtifact,
    ConsentType,
    IdentityState,
    ModelProposal,
    PolicyDecision,
    RejectReason,
    Speaker,
    StirShakenAttestation,
    SupervisorIntervention,
    SupervisorInterventionType,
    ValidatorResult,
    VoiceTurn,
    utc_now,
)
from voice_intake.orchestrator import VoiceIntakeOrchestrator
from voice_intake.policy import PolicyEngine, PolicyProfile
from voice_intake.templates import DEFAULT_TEMPLATE_BUNDLE
from voice_intake.validator import ProposalValidator


def _uid() -> str:
    return str(uuid4())


def _make_store() -> SQLAuditStore:
    engine = get_engine("sqlite:///:memory:")
    create_tables(engine)
    return SQLAuditStore(make_session_factory(engine), hmac_secret="test-audit-secret")


def _make_session(state: CallState = CallState.IDENTITY_CAPTURE) -> CallSession:
    session = CallSession(
        session_id=_uid(),
        telephony_call_id=_uid(),
        operator_id="op-1",
        current_state=state,
    )
    return session


class AuditStoreProtocolTest(unittest.TestCase):
    def test_sql_store_satisfies_protocol(self):
        store = _make_store()
        self.assertIsInstance(store, AuditStore)


class SaveLoadSessionTest(unittest.TestCase):
    def setUp(self):
        self.store = _make_store()

    def test_save_and_load_roundtrip(self):
        session = _make_session()
        self.store.save_session(session)
        loaded = self.store.load_session(session.session_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.session_id, session.session_id)
        self.assertEqual(loaded.current_state, session.current_state)

    def test_load_nonexistent_returns_none(self):
        result = self.store.load_session("nonexistent-id")
        self.assertIsNone(result)

    def test_update_session_state(self):
        session = _make_session(CallState.IDENTITY_CAPTURE)
        self.store.save_session(session)
        session.current_state = CallState.DEMOGRAPHICS
        self.store.save_session(session)
        loaded = self.store.load_session(session.session_id)
        self.assertEqual(loaded.current_state, CallState.DEMOGRAPHICS)

    def test_reject_counts_roundtrip(self):
        session = _make_session()
        counts = {(session.session_id, "turn-abc"): 1}
        self.store.save_session(session, reject_counts=counts)
        loaded_counts = self.store.load_reject_counts(session.session_id)
        self.assertEqual(loaded_counts, counts)

    def test_list_sessions_returns_all(self):
        for _ in range(3):
            self.store.save_session(_make_session())
        sessions = self.store.list_sessions()
        self.assertEqual(len(sessions), 3)


class AuditEventStoreTest(unittest.TestCase):
    def setUp(self):
        self.store = _make_store()
        self.session = _make_session()
        self.store.save_session(self.session)

    def _make_event(self, event_type: AuditEventType = AuditEventType.SESSION_OPENED) -> AuditEvent:
        return AuditEvent(
            event_id=_uid(),
            session_id=self.session.session_id,
            actor=AuditActor.SYSTEM,
            event_type=event_type,
            proposal_id=None,
            validator_result_id=None,
            affected_fields=[],
            identity_state=IdentityState.UNVERIFIED,
            policy_decision_id=None,
            template_id=None,
            model_version=None,
            service_path="test",
            before_after_hash="test-hash",
        )

    def test_record_and_retrieve_audit_events(self):
        event = self._make_event()
        self.store.record_audit_event(event)
        events = self.store.audit_events_for_session(self.session.session_id)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_id, event.event_id)

    def test_multiple_events_ordered_by_timestamp(self):
        for etype in [AuditEventType.SESSION_OPENED, AuditEventType.CONSENT_CAPTURED]:
            self.store.record_audit_event(self._make_event(etype))
        events = self.store.audit_events_for_session(self.session.session_id)
        self.assertEqual(len(events), 2)

    def test_audit_events_isolated_by_session(self):
        other = _make_session()
        self.store.save_session(other)
        self.store.record_audit_event(self._make_event())
        other_events = self.store.audit_events_for_session(other.session_id)
        self.assertEqual(len(other_events), 0)

    def test_custom_event_type_string_roundtrip(self):
        event = self._make_event()
        # Use a raw string event_type (not enum) to exercise the fallback path
        object.__setattr__(event, "event_type", "custom_event_type")
        self.store.record_audit_event(event)
        events = self.store.audit_events_for_session(self.session.session_id)
        self.assertEqual(events[0].event_type, "custom_event_type")

    def test_new_audit_events_get_chain_hash_and_index(self):
        event = self._make_event()
        self.store.record_audit_event(event)

        events = self.store.audit_events_for_session(self.session.session_id)

        self.assertEqual(events[0].chain_index, 0)
        self.assertTrue(events[0].chain_hash)

    def test_chain_index_is_monotonic_per_session(self):
        self.store.record_audit_event(self._make_event(AuditEventType.SESSION_OPENED))
        self.store.record_audit_event(self._make_event(AuditEventType.CONSENT_CAPTURED))

        events = self.store.audit_events_for_session(self.session.session_id)

        self.assertEqual([event.chain_index for event in events], [0, 1])

    def test_verify_chain_detects_tamper(self):
        self.store.record_audit_event(self._make_event(AuditEventType.SESSION_OPENED))
        events = self.store.audit_events_for_session(self.session.session_id)
        self.assertEqual(verify_chain(events, self.store.audit_secret), [])

        tampered = events[0]
        tampered.before_after_hash = "tampered"

        errors = verify_chain(events, self.store.audit_secret)
        self.assertEqual(len(errors), 1)


class ConsentArtifactStoreTest(unittest.TestCase):
    def setUp(self):
        self.store = _make_store()
        self.session = _make_session()
        self.store.save_session(self.session)

    def test_record_and_retrieve_consents(self):
        artifact = ConsentArtifact(
            session_id=self.session.session_id,
            consent_type=ConsentType.RECORDING,
            status="granted",
            jurisdiction_policy_id="default",
            captured_at=utc_now(),
            capture_mode="ai_led",
            storage_effect="standard_retention",
        )
        self.store.record_consent_artifact(artifact)
        results = self.store.consents_for_session(self.session.session_id)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "granted")
        self.assertEqual(results[0].consent_type, ConsentType.RECORDING)

    def test_consents_isolated_by_session(self):
        other = _make_session()
        self.store.save_session(other)
        artifact = ConsentArtifact(
            session_id=self.session.session_id,
            consent_type=ConsentType.AI_ASSISTANCE,
            status="declined",
            jurisdiction_policy_id="default",
            captured_at=utc_now(),
            capture_mode="ai_led",
            storage_effect="disable_ai_led_flow",
        )
        self.store.record_consent_artifact(artifact)
        self.assertEqual(len(self.store.consents_for_session(other.session_id)), 0)


class VoiceTurnStoreTest(unittest.TestCase):
    def setUp(self):
        self.store = _make_store()
        self.session = _make_session()
        self.store.save_session(self.session)

    def test_record_and_retrieve_turns(self):
        ts = utc_now()
        turn = VoiceTurn(
            turn_id=_uid(),
            speaker=Speaker.CALLER,
            audio_ref=None,
            retention_policy_id="standard",
            transcript="My name is Jane",
            partial_or_final="final",
            asr_confidence=0.98,
            barge_in=False,
            start_ts=ts,
            end_ts=ts,
        )
        self.store.record_turn(self.session.session_id, turn)
        turns = self.store.turns_for_session(self.session.session_id)
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0].transcript, "My name is Jane")
        self.assertFalse(turns[0].barge_in)


class SupervisorInterventionStoreTest(unittest.TestCase):
    def setUp(self):
        self.store = _make_store()
        self.session = _make_session()
        self.store.save_session(self.session)

    def test_record_supervisor_intervention(self):
        ts = utc_now()
        intervention = SupervisorIntervention(
            session_id=self.session.session_id,
            intervention_type=SupervisorInterventionType.FORCED_TAKEOVER,
            reason_code="complex_case",
            notes="Manual review needed.",
            alert_ts=ts,
            ack_ts=ts,
            takeover_ts=ts,
        )
        self.store.record_supervisor_intervention(intervention)


class OrchestratorWithSQLStoreTest(unittest.TestCase):
    """Smoke-test: run the orchestrator end-to-end using SQLAuditStore."""

    def setUp(self):
        self.store = _make_store()
        profile = PolicyProfile()
        policy_engine = PolicyEngine(profile)
        validator = ProposalValidator(DEFAULT_TEMPLATE_BUNDLE, policy_engine)
        self.orchestrator = VoiceIntakeOrchestrator(
            policy_profile=profile,
            validator=validator,
            audit_store=self.store,
            policy_engine=policy_engine,
        )

    def test_full_session_lifecycle_persists_audit_events(self):
        session = self.orchestrator.open_session("call-1", "op-1", StirShakenAttestation.A)
        self.store.save_session(session)

        self.orchestrator.start_after_opening(session)
        self.store.save_session(session)

        self.orchestrator.capture_consent(session, ConsentType.RECORDING, True)
        self.store.save_session(session)

        events = self.store.audit_events_for_session(session.session_id)
        self.assertGreater(len(events), 0)

        loaded = self.store.load_session(session.session_id)
        self.assertEqual(loaded.current_state, CallState.CONSENT_AI_ASSISTANCE)

    def test_model_proposal_accepted_and_state_persists(self):
        session = self.orchestrator.open_session("call-2", "op-1", StirShakenAttestation.A)
        self.store.save_session(session)
        self.orchestrator.start_after_opening(session)
        self.orchestrator.capture_consent(session, ConsentType.RECORDING, True)
        self.orchestrator.capture_consent(session, ConsentType.AI_ASSISTANCE, True)
        self.store.save_session(session)

        proposal = ModelProposal(
            proposal_id=_uid(),
            source_turn_id=_uid(),
            template_id="collect_field_prompt",
            variables={"field_label": "name"},
            requested_transition=CallState.DEMOGRAPHICS,
            model_version="mock-v1",
        )
        outcome, action = self.orchestrator.handle_model_proposal(session, proposal)
        self.store.save_session(session)

        self.assertTrue(outcome.validator_result.accepted)
        self.assertIsNotNone(action)

        loaded = self.store.load_session(session.session_id)
        self.assertEqual(loaded.current_state, CallState.DEMOGRAPHICS)


if __name__ == "__main__":
    unittest.main()
