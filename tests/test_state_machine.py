from __future__ import annotations

import unittest

from voice_intake.models import CallState
from voice_intake.state_machine import allowed_targets, can_transition


class StateMachineTest(unittest.TestCase):
    def test_opening_has_only_declared_targets(self) -> None:
        self.assertTrue(can_transition(CallState.OPENING, CallState.CONSENT_RECORDING))
        self.assertTrue(can_transition(CallState.OPENING, CallState.CONSENT_AI_ASSISTANCE))
        self.assertFalse(can_transition(CallState.OPENING, CallState.INSURANCE))

    def test_close_is_terminal(self) -> None:
        self.assertEqual(allowed_targets(CallState.CLOSE), frozenset())

    def test_demo_transitions_added_without_removing_existing(self) -> None:
        """Phase 4.5 regression: demo-flow transitions were added without
        removing any pre-existing transitions."""
        # OPENING → REASON_FOR_VISIT added for demo flow (skips consent)
        self.assertTrue(can_transition(CallState.OPENING, CallState.REASON_FOR_VISIT))
        # Pre-existing OPENING → CONSENT_* transitions must be preserved
        self.assertTrue(can_transition(CallState.OPENING, CallState.CONSENT_RECORDING))
        self.assertTrue(can_transition(CallState.OPENING, CallState.CONSENT_AI_ASSISTANCE))

        # REASON_FOR_VISIT → IDENTITY_CAPTURE added (demo: reason → name)
        self.assertTrue(can_transition(CallState.REASON_FOR_VISIT, CallState.IDENTITY_CAPTURE))
        # Pre-existing REASON_FOR_VISIT → READBACK kept
        self.assertTrue(can_transition(CallState.REASON_FOR_VISIT, CallState.READBACK))

        # INSURANCE → READBACK added (demo: insurance → confirmation)
        self.assertTrue(can_transition(CallState.INSURANCE, CallState.READBACK))
        # Pre-existing INSURANCE → REASON_FOR_VISIT kept
        self.assertTrue(can_transition(CallState.INSURANCE, CallState.REASON_FOR_VISIT))

        # Self-loops on all field-collection and readback states
        self.assertTrue(can_transition(CallState.REASON_FOR_VISIT, CallState.REASON_FOR_VISIT))
        self.assertTrue(can_transition(CallState.IDENTITY_CAPTURE, CallState.IDENTITY_CAPTURE))
        self.assertTrue(can_transition(CallState.DEMOGRAPHICS, CallState.DEMOGRAPHICS))
        self.assertTrue(can_transition(CallState.INSURANCE, CallState.INSURANCE))
        self.assertTrue(can_transition(CallState.READBACK, CallState.READBACK))

    def test_opening_self_loop_for_meta_questions(self) -> None:
        """C4.4: OPENING -> OPENING permits answering meta/help questions
        without leaving the opening state."""
        self.assertTrue(can_transition(CallState.OPENING, CallState.OPENING))


if __name__ == "__main__":
    unittest.main()

