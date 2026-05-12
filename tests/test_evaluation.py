from __future__ import annotations

import unittest

from voice_intake.evaluation import (
    PartitionMetric,
    PilotObservation,
    evaluate_asr_partitions,
    evaluate_pilot_gate,
    validate_transition_history,
)
from voice_intake.models import CallState


class EvaluationTest(unittest.TestCase):
    def test_asr_disparity_gate_fails_above_threshold(self) -> None:
        result = evaluate_asr_partitions(
            [
                PartitionMetric("northeast", wer=10.0, critical_slot_digit_accuracy=99.0),
                PartitionMetric("non_native", wer=16.0, critical_slot_digit_accuracy=93.0),
            ]
        )
        self.assertFalse(result.passed)
        self.assertEqual(len(result.errors), 2)

    def test_pilot_gate_requires_all_conditions(self) -> None:
        result = evaluate_pilot_gate(
            PilotObservation(
                completed_calls=500,
                phi_leak_count=0,
                confirmation_rate=1.0,
                latency_met=True,
                transfer_rate_in_band=True,
                incident_drill_passed=True,
            )
        )
        self.assertTrue(result.passed)

    def test_transition_history_validation(self) -> None:
        valid = [
            CallState.OPENING,
            CallState.CONSENT_RECORDING,
            CallState.CONSENT_AI_ASSISTANCE,
            CallState.IDENTITY_CAPTURE,
            CallState.DEMOGRAPHICS,
        ]
        invalid = [CallState.OPENING, CallState.INSURANCE]
        self.assertTrue(validate_transition_history(valid))
        self.assertFalse(validate_transition_history(invalid))


if __name__ == "__main__":
    unittest.main()

