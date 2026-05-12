from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone

from voice_intake.models import (
    CallSession,
    CallState,
    FieldCandidate,
    Speaker,
    VerificationStatus,
    VoiceTurn,
)
from voice_intake.prompting import build_model_prompt
from voice_intake.templates import DEFAULT_TEMPLATE_BUNDLE


class PromptingTest(unittest.TestCase):
    def test_prompt_redacts_raw_values_and_uses_masked_confirmation(self) -> None:
        session = CallSession(session_id="s1", telephony_call_id="c1", operator_id="o1")
        session.current_state = CallState.READBACK
        candidate = FieldCandidate(
            field_name="member_id",
            candidate_value="123456789",
            confidence=0.99,
            verification_status=VerificationStatus.PENDING_CONFIRMATION,
            source_turn_ids=["t1"],
            sensitivity_class="phi",
        )
        turn = VoiceTurn(
            turn_id="t1",
            speaker=Speaker.CALLER,
            audio_ref=None,
            retention_policy_id="default",
            transcript="My member ID is 123456789",
            partial_or_final="final",
            asr_confidence=0.98,
            barge_in=False,
            start_ts=datetime.now(timezone.utc),
            end_ts=datetime.now(timezone.utc),
        )
        prompt = build_model_prompt(
            session=session,
            recent_turns=[turn],
            field_candidates=[candidate],
            available_templates=[DEFAULT_TEMPLATE_BUNDLE.templates["readback_confirmation"]],
            confirmation_field=candidate,
        )
        serialized = json.dumps(prompt)
        self.assertNotIn("123456789", serialized)
        self.assertIn("*****6789", serialized)


class AllowedTransitionsTest(unittest.TestCase):
    """C4.4: build_model_prompt serializes allowed_transitions for the LLM."""

    def _session(self, state: CallState) -> CallSession:
        s = CallSession(session_id="s1", telephony_call_id="c1", operator_id="o1")
        s.current_state = state
        return s

    def test_allowed_transitions_present_when_provided(self):
        from voice_intake.policy import demo_policy_profile, model_allowed_transitions
        session = self._session(CallState.OPENING)
        transitions = model_allowed_transitions(demo_policy_profile(), CallState.OPENING)
        prompt = build_model_prompt(
            session=session,
            recent_turns=[],
            field_candidates=[],
            available_templates=[],
            allowed_transitions=transitions,
        )
        self.assertIn("allowed_transitions", prompt)
        # Must serialize as state-name strings, not enums
        for value in prompt["allowed_transitions"]:
            self.assertIsInstance(value, str)
        self.assertIn("opening", prompt["allowed_transitions"])
        self.assertIn("reason_for_visit", prompt["allowed_transitions"])
        self.assertNotIn("consent_recording", prompt["allowed_transitions"])
        self.assertNotIn("consent_ai_assistance", prompt["allowed_transitions"])

    def test_allowed_transitions_absent_by_default(self):
        """Backward compat: callers that don't pass allowed_transitions don't get the key."""
        session = self._session(CallState.OPENING)
        prompt = build_model_prompt(
            session=session,
            recent_turns=[],
            field_candidates=[],
            available_templates=[],
        )
        self.assertNotIn("allowed_transitions", prompt)


class ModelFacingTemplateTest(unittest.TestCase):
    """C4.4: hold_message is excluded from model-facing templates."""

    def test_hold_message_is_not_model_facing(self):
        bundle = DEFAULT_TEMPLATE_BUNDLE
        self.assertFalse(bundle.templates["hold_message"].model_facing)

    def test_hold_message_is_still_in_bundle(self):
        """Validator must continue to accept hold_message proposals from system paths."""
        bundle = DEFAULT_TEMPLATE_BUNDLE
        self.assertIn("hold_message", bundle.templates)

    def test_other_templates_default_to_model_facing(self):
        bundle = DEFAULT_TEMPLATE_BUNDLE
        for tid, template in bundle.templates.items():
            if tid == "hold_message":
                continue
            self.assertTrue(
                template.model_facing,
                f"Template {tid} unexpectedly has model_facing=False",
            )


if __name__ == "__main__":
    unittest.main()
