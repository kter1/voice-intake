from __future__ import annotations

import unittest

from voice_intake.audit import InMemoryAuditStore
from voice_intake.models import (
    CallDisposition,
    CallMode,
    CallState,
    ConsentType,
    ManualModeProfile,
    StirShakenAttestation,
)
from voice_intake.orchestrator import VoiceIntakeOrchestrator
from voice_intake.policy import PolicyEngine, PolicyProfile
from voice_intake.templates import DEFAULT_TEMPLATE_BUNDLE
from voice_intake.validator import ProposalValidator


class PolicyAndOrchestratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = PolicyProfile(
            require_recording_consent=True,
            require_ai_assistance_consent=True,
            allow_ai_without_recording=True,
        )
        self.audit_store = InMemoryAuditStore()
        self.engine = PolicyEngine(self.profile)
        self.validator = ProposalValidator(DEFAULT_TEMPLATE_BUNDLE, self.engine)
        self.orchestrator = VoiceIntakeOrchestrator(
            policy_profile=self.profile,
            validator=self.validator,
            audit_store=self.audit_store,
        )

    def test_recording_decline_can_continue_to_ai_consent(self) -> None:
        session = self.orchestrator.open_session("call-1", "op-1", StirShakenAttestation.B)
        session.current_state = CallState.CONSENT_RECORDING
        next_state = self.orchestrator.capture_consent(
            session=session, consent_type=ConsentType.RECORDING, granted=False
        )
        self.assertEqual(next_state, CallState.CONSENT_AI_ASSISTANCE)
        artifact = self.audit_store.consents_for_session(session.session_id)[0]
        self.assertIn("suppress_transcript_persistence", artifact.storage_effect)
        self.assertEqual(session.mode, CallMode.AI_LED)

    def test_ai_decline_after_recording_decline_forces_consent_declined(self) -> None:
        session = self.orchestrator.open_session("call-2", "op-1", StirShakenAttestation.B)
        session.current_state = CallState.CONSENT_RECORDING
        self.orchestrator.capture_consent(
            session=session, consent_type=ConsentType.RECORDING, granted=False
        )
        session.current_state = CallState.CONSENT_AI_ASSISTANCE
        next_state = self.orchestrator.capture_consent(
            session=session, consent_type=ConsentType.AI_ASSISTANCE, granted=False
        )
        self.assertEqual(next_state, CallState.CONSENT_DECLINED)
        self.assertEqual(session.mode, CallMode.MANUAL)

    def test_manual_mode_requires_storage_suppression_for_disabled_transcription(self) -> None:
        restricted_profile = PolicyProfile(
            manual_mode_profile=ManualModeProfile(transcription_mode="disabled")
        )
        restricted_engine = PolicyEngine(restricted_profile)
        validator = ProposalValidator(DEFAULT_TEMPLATE_BUNDLE, restricted_engine)
        orchestrator = VoiceIntakeOrchestrator(
            policy_profile=restricted_profile,
            validator=validator,
            audit_store=InMemoryAuditStore(),
        )
        session = orchestrator.open_session("call-3", "op-1", StirShakenAttestation.UNKNOWN)
        with self.assertRaises(ValueError):
            orchestrator.enter_manual_mode(session)

    def test_emergency_abandonment_sets_distinct_disposition(self) -> None:
        session = self.orchestrator.open_session("call-4", "op-1", StirShakenAttestation.C)
        session.current_state = CallState.EMERGENCY_EXIT
        self.orchestrator.abandon_call(session)
        self.assertEqual(session.disposition, CallDisposition.EMERGENCY_ABANDONED)
        self.assertTrue(
            any(event.event_type == "emergency_abandoned" for event in self.audit_store.audit_events)
        )

    def test_model_timeout_preserves_buffered_audio_reference(self) -> None:
        session = self.orchestrator.open_session("call-5", "op-1", StirShakenAttestation.UNKNOWN)
        self.orchestrator.handle_model_timeout(session, buffered_audio_ref="audio://buffered")
        self.assertEqual(session.current_state, CallState.HUMAN_TAKEOVER)
        self.assertEqual(session.metadata["buffered_audio_ref"], "audio://buffered")


class ModelAllowedTransitionsTest(unittest.TestCase):
    """C4.4: model_allowed_transitions filters policy-disallowed targets."""

    def test_demo_profile_from_opening_excludes_consent_states(self):
        from voice_intake.policy import demo_policy_profile, model_allowed_transitions
        targets = model_allowed_transitions(demo_policy_profile(), CallState.OPENING)
        self.assertNotIn(CallState.CONSENT_RECORDING, targets)
        self.assertNotIn(CallState.CONSENT_AI_ASSISTANCE, targets)
        self.assertNotIn(CallState.CONSENT_DECLINED, targets)
        self.assertNotIn(CallState.MANUAL_MODE, targets)

    def test_demo_profile_from_opening_includes_demo_targets(self):
        from voice_intake.policy import demo_policy_profile, model_allowed_transitions
        targets = model_allowed_transitions(demo_policy_profile(), CallState.OPENING)
        self.assertIn(CallState.OPENING, targets)  # self-loop for meta questions
        self.assertIn(CallState.REASON_FOR_VISIT, targets)
        self.assertIn(CallState.HUMAN_TAKEOVER, targets)
        self.assertIn(CallState.EMERGENCY_EXIT, targets)

    def test_default_profile_unchanged(self):
        """Regression guard: default policy must still see consent transitions."""
        from voice_intake.policy import model_allowed_transitions
        from voice_intake.state_machine import allowed_targets

        targets = model_allowed_transitions(PolicyProfile(), CallState.OPENING)
        # Default policy: targets must equal raw state-machine targets exactly.
        self.assertEqual(targets, allowed_targets(CallState.OPENING))
        self.assertIn(CallState.CONSENT_RECORDING, targets)
        self.assertIn(CallState.CONSENT_AI_ASSISTANCE, targets)

    def test_demo_profile_from_consent_recording_still_excludes_consent_targets(self):
        """Even from a consent state, demo policy hides further consent targets."""
        from voice_intake.policy import demo_policy_profile, model_allowed_transitions
        targets = model_allowed_transitions(
            demo_policy_profile(), CallState.CONSENT_RECORDING
        )
        self.assertNotIn(CallState.CONSENT_AI_ASSISTANCE, targets)
        self.assertNotIn(CallState.CONSENT_DECLINED, targets)


if __name__ == "__main__":
    unittest.main()

