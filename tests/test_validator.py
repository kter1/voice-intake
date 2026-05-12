from __future__ import annotations

import unittest
from uuid import uuid4

from voice_intake.audit import InMemoryAuditStore
from voice_intake.models import (
    CallMode,
    CallSession,
    CallState,
    ModelProposal,
    RejectReason,
    StirShakenAttestation,
)
from voice_intake.orchestrator import VoiceIntakeOrchestrator
from voice_intake.policy import PolicyEngine, PolicyProfile
from voice_intake.templates import DEFAULT_TEMPLATE_BUNDLE
from voice_intake.validator import ProposalValidator


def make_session(state: CallState, mode: CallMode = CallMode.AI_LED) -> CallSession:
    return CallSession(
        session_id=str(uuid4()),
        telephony_call_id="call-1",
        operator_id="op-1",
        current_state=state,
        mode=mode,
        policy_profile_id="default",
        template_bundle_version=DEFAULT_TEMPLATE_BUNDLE.version,
        stir_shaken_attestation=StirShakenAttestation.A,
    )


class ValidatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = PolicyProfile()
        self.engine = PolicyEngine(self.policy)
        self.validator = ProposalValidator(DEFAULT_TEMPLATE_BUNDLE, self.engine)

    def test_valid_collect_prompt_is_accepted(self) -> None:
        session = make_session(CallState.IDENTITY_CAPTURE)
        proposal = ModelProposal(
            proposal_id="proposal-1",
            source_turn_id="turn-1",
            template_id="collect_field_prompt",
            variables={"field_label": "date of birth"},
            requested_transition=CallState.DEMOGRAPHICS,
            model_version="model-v1",
        )
        outcome = self.validator.validate(session, proposal)
        self.assertTrue(outcome.validator_result.accepted)
        self.assertIsNone(outcome.validator_result.reject_reason)
        self.assertIsNotNone(outcome.policy_decision)

    def test_invalid_transition_is_rejected(self) -> None:
        session = make_session(CallState.IDENTITY_CAPTURE)
        proposal = ModelProposal(
            proposal_id="proposal-2",
            source_turn_id="turn-2",
            template_id="collect_field_prompt",
            variables={"field_label": "date of birth"},
            requested_transition=CallState.CLOSE,
            model_version="model-v1",
        )
        outcome = self.validator.validate(session, proposal)
        self.assertFalse(outcome.validator_result.accepted)
        self.assertEqual(outcome.validator_result.reject_reason, RejectReason.DISALLOWED_TRANSITION)

    def test_masked_value_requirement_is_enforced(self) -> None:
        session = make_session(CallState.READBACK)
        proposal = ModelProposal(
            proposal_id="proposal-3",
            source_turn_id="turn-3",
            template_id="readback_confirmation",
            variables={"field_label": "member id", "masked_value": "123456789"},
            requested_transition=CallState.DISPOSITION,
            model_version="model-v1",
        )
        outcome = self.validator.validate(session, proposal)
        self.assertFalse(outcome.validator_result.accepted)
        self.assertEqual(outcome.validator_result.reject_reason, RejectReason.OUT_OF_VOCABULARY_PHI)

    def test_policy_denial_in_manual_mode(self) -> None:
        session = make_session(CallState.READBACK, mode=CallMode.MANUAL)
        proposal = ModelProposal(
            proposal_id="proposal-4",
            source_turn_id="turn-4",
            template_id="readback_confirmation",
            variables={"field_label": "member id", "masked_value": "*****6789"},
            requested_transition=CallState.DISPOSITION,
            model_version="model-v1",
        )
        outcome = self.validator.validate(session, proposal)
        self.assertFalse(outcome.validator_result.accepted)
        self.assertEqual(outcome.validator_result.reject_reason, RejectReason.POLICY_DENIED)

    def test_double_reject_forces_human_takeover(self) -> None:
        audit_store = InMemoryAuditStore()
        orchestrator = VoiceIntakeOrchestrator(
            policy_profile=self.policy,
            validator=self.validator,
            audit_store=audit_store,
        )
        session = make_session(CallState.IDENTITY_CAPTURE)
        proposal = ModelProposal(
            proposal_id="proposal-5",
            source_turn_id="turn-5",
            template_id="collect_field_prompt",
            variables={"field_label": "date of birth"},
            requested_transition=CallState.CLOSE,
            model_version="model-v1",
        )
        orchestrator.handle_model_proposal(session, proposal)
        self.assertEqual(session.current_state, CallState.IDENTITY_CAPTURE)
        orchestrator.handle_model_proposal(session, proposal)
        self.assertEqual(session.current_state, CallState.HUMAN_TAKEOVER)
        self.assertTrue(
            any(event.event_type == "validator_double_reject" for event in audit_store.audit_events)
        )

    def test_double_reject_across_different_proposals_in_same_state(self) -> None:
        """Two separate proposals in the same state escalate (state-keyed counter)."""
        audit_store = InMemoryAuditStore()
        orchestrator = VoiceIntakeOrchestrator(
            policy_profile=self.policy,
            validator=self.validator,
            audit_store=audit_store,
        )
        session = make_session(CallState.IDENTITY_CAPTURE)
        # Two different proposals (different proposal_id AND source_turn_id), both invalid.
        bad1 = ModelProposal(
            proposal_id="p-bad-1",
            source_turn_id="turn-A",
            template_id="collect_field_prompt",
            variables={"field_label": "date of birth"},
            requested_transition=CallState.CLOSE,  # disallowed from IDENTITY_CAPTURE
            model_version="model-v1",
        )
        bad2 = ModelProposal(
            proposal_id="p-bad-2",
            source_turn_id="turn-B",  # different turn id
            template_id="collect_field_prompt",
            variables={"field_label": "date of birth"},
            requested_transition=CallState.CLOSE,
            model_version="model-v1",
        )
        orchestrator.handle_model_proposal(session, bad1)
        self.assertEqual(session.current_state, CallState.IDENTITY_CAPTURE)
        orchestrator.handle_model_proposal(session, bad2)
        self.assertEqual(session.current_state, CallState.HUMAN_TAKEOVER)

    def test_emergency_redirect_accepted_from_normal_state(self) -> None:
        """emergency_redirect must be valid from any normal source state (not just EMERGENCY_EXIT)."""
        session = make_session(CallState.IDENTITY_CAPTURE)
        proposal = ModelProposal(
            proposal_id="proposal-emerg",
            source_turn_id="turn-emerg",
            template_id="emergency_redirect",
            variables={},
            requested_transition=CallState.EMERGENCY_EXIT,
            model_version="model-v1",
        )
        outcome = self.validator.validate(session, proposal)
        self.assertTrue(outcome.validator_result.accepted)
        self.assertIsNone(outcome.validator_result.reject_reason)

    def test_closing_prompt_accepted_from_disposition(self) -> None:
        """closing_prompt must be valid from DISPOSITION (transitioning to CLOSE)."""
        session = make_session(CallState.DISPOSITION)
        proposal = ModelProposal(
            proposal_id="proposal-close",
            source_turn_id="turn-close",
            template_id="closing_prompt",
            variables={},
            requested_transition=CallState.CLOSE,
            model_version="model-v1",
        )
        outcome = self.validator.validate(session, proposal)
        self.assertTrue(outcome.validator_result.accepted)
        self.assertIsNone(outcome.validator_result.reject_reason)


if __name__ == "__main__":
    unittest.main()

