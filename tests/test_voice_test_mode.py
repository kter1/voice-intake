import json
import unittest

from voice_intake.models import CallMode, CallState
from voice_intake.testing import ScriptedCallRunner
from voice_intake.testing.fixtures.consent_declined import (
    BOTH_CONSENTS_DECLINED,
    CONSENT_RECORDING_DECLINED,
)
from voice_intake.testing.fixtures.double_reject import DOUBLE_REJECT
from voice_intake.testing.fixtures.emergency_exit import EMERGENCY_EXIT
from voice_intake.testing.fixtures.happy_path import HAPPY_PATH
from voice_intake.testing.fixtures.supervisor_takeover import SUPERVISOR_TAKEOVER


class HappyPathTest(unittest.TestCase):
    def setUp(self):
        self.trace = ScriptedCallRunner(HAPPY_PATH).run()

    def test_all_steps_pass(self):
        failures = [
            f"step {r.step_index}: {r.failure_reason}"
            for r in self.trace.step_results
            if not r.passed
        ]
        self.assertEqual(failures, [], msg="\n".join(failures))

    def test_final_state_is_close(self):
        self.assertEqual(self.trace.session.current_state, CallState.CLOSE)

    def test_audit_events_recorded(self):
        self.assertGreater(self.trace.audit_event_count, 0)

    def test_no_raw_phi_in_prompts(self):
        runner = ScriptedCallRunner(HAPPY_PATH)
        runner.run()
        hits = runner.prompt_contains_raw_phi()
        self.assertEqual(hits, [], msg=f"Raw PHI found in prompt snapshots: {hits}")

    def test_voice_actions_have_correct_templates(self):
        llm_steps = [r for r in self.trace.step_results if r.emitted_action is not None]
        templates_used = [r.emitted_action.template_id for r in llm_steps]
        self.assertIn("collect_field_prompt", templates_used)
        self.assertIn("readback_confirmation", templates_used)
        self.assertIn("hold_message", templates_used)

    def test_consent_steps_have_no_voice_action(self):
        consent_results = self.trace.step_results[:2]
        for r in consent_results:
            self.assertIsNone(r.emitted_action)

    def test_prompts_include_current_state(self):
        runner = ScriptedCallRunner(HAPPY_PATH)
        runner.run()
        for snapshot in runner._prompt_snapshots:
            self.assertIn("current_state", snapshot)

    def test_prompts_include_available_templates(self):
        runner = ScriptedCallRunner(HAPPY_PATH)
        runner.run()
        for snapshot in runner._prompt_snapshots:
            self.assertIn("available_templates", snapshot)


class ConsentDeclinedTest(unittest.TestCase):
    def test_recording_declined_ai_granted_proceeds(self):
        trace = ScriptedCallRunner(CONSENT_RECORDING_DECLINED).run()
        failures = [
            f"step {r.step_index}: {r.failure_reason}"
            for r in trace.step_results
            if not r.passed
        ]
        self.assertEqual(failures, [], msg="\n".join(failures))
        self.assertEqual(trace.session.current_state, CallState.DEMOGRAPHICS)

    def test_both_declined_lands_in_consent_declined(self):
        trace = ScriptedCallRunner(BOTH_CONSENTS_DECLINED).run()
        self.assertTrue(trace.all_passed)
        self.assertEqual(trace.session.current_state, CallState.CONSENT_DECLINED)

    def test_consent_artifacts_recorded(self):
        runner = ScriptedCallRunner(CONSENT_RECORDING_DECLINED)
        trace = runner.run()
        artifacts = runner.audit_store.consent_artifacts
        self.assertEqual(len(artifacts), 2)
        statuses = {a.consent_type.value: a.status for a in artifacts}
        self.assertEqual(statuses["recording_consent"], "declined")
        self.assertEqual(statuses["ai_assistance_consent"], "granted")

    def test_storage_effect_suppresses_on_recording_decline(self):
        runner = ScriptedCallRunner(CONSENT_RECORDING_DECLINED)
        runner.run()
        recording_artifact = next(
            a for a in runner.audit_store.consent_artifacts
            if a.consent_type.value == "recording_consent"
        )
        self.assertIn("suppress", recording_artifact.storage_effect)


class EmergencyExitTest(unittest.TestCase):
    def test_all_steps_pass(self):
        trace = ScriptedCallRunner(EMERGENCY_EXIT).run()
        failures = [
            f"step {r.step_index}: {r.failure_reason}"
            for r in trace.step_results
            if not r.passed
        ]
        self.assertEqual(failures, [], msg="\n".join(failures))

    def test_emergency_trigger_lands_in_emergency_exit(self):
        trace = ScriptedCallRunner(EMERGENCY_EXIT).run()
        # Second-to-last step transitions to EMERGENCY_EXIT
        trigger_result = trace.step_results[-2]
        self.assertEqual(trigger_result.actual_state, CallState.EMERGENCY_EXIT)

    def test_emergency_redirect_action_emitted(self):
        trace = ScriptedCallRunner(EMERGENCY_EXIT).run()
        # Last step plays emergency_redirect from EMERGENCY_EXIT
        redirect_result = trace.step_results[-1]
        self.assertIsNotNone(redirect_result.emitted_action)
        self.assertEqual(redirect_result.emitted_action.template_id, "emergency_redirect")

    def test_emergency_action_is_not_interruptible(self):
        trace = ScriptedCallRunner(EMERGENCY_EXIT).run()
        redirect_result = trace.step_results[-1]
        self.assertFalse(redirect_result.emitted_action.interruptible)

    def test_final_state_is_human_takeover(self):
        trace = ScriptedCallRunner(EMERGENCY_EXIT).run()
        self.assertEqual(trace.session.current_state, CallState.HUMAN_TAKEOVER)


class DoubleRejectTest(unittest.TestCase):
    def test_double_reject_forces_human_takeover(self):
        trace = ScriptedCallRunner(DOUBLE_REJECT).run()
        self.assertTrue(trace.all_passed)
        self.assertEqual(trace.session.current_state, CallState.HUMAN_TAKEOVER)

    def test_both_rejection_steps_have_no_action(self):
        trace = ScriptedCallRunner(DOUBLE_REJECT).run()
        llm_results = [r for r in trace.step_results if r.validation_outcome is not None]
        for r in llm_results:
            self.assertIsNone(r.emitted_action)

    def test_double_reject_audit_event_recorded(self):
        runner = ScriptedCallRunner(DOUBLE_REJECT)
        runner.run()
        event_types = [e.event_type for e in runner.audit_store.audit_events]
        self.assertIn("validator_double_reject", event_types)

    def test_mode_becomes_human_takeover(self):
        trace = ScriptedCallRunner(DOUBLE_REJECT).run()
        self.assertEqual(trace.session.mode, CallMode.HUMAN_TAKEOVER)


class SupervisorTakeoverTest(unittest.TestCase):
    def test_forced_takeover_transitions_to_human_takeover(self):
        trace = ScriptedCallRunner(SUPERVISOR_TAKEOVER).run()
        self.assertTrue(trace.all_passed)
        self.assertEqual(trace.session.current_state, CallState.HUMAN_TAKEOVER)

    def test_supervisor_intervention_recorded_in_audit(self):
        runner = ScriptedCallRunner(SUPERVISOR_TAKEOVER)
        runner.run()
        self.assertEqual(len(runner.audit_store.supervisor_interventions), 1)
        intervention = runner.audit_store.supervisor_interventions[0]
        self.assertEqual(intervention.intervention_type.value, "forced_takeover")
        self.assertEqual(intervention.reason_code, "complex_case")

    def test_mode_becomes_human_takeover_after_forced_takeover(self):
        trace = ScriptedCallRunner(SUPERVISOR_TAKEOVER).run()
        self.assertEqual(trace.session.mode, CallMode.HUMAN_TAKEOVER)

    def test_prior_steps_before_takeover_passed(self):
        trace = ScriptedCallRunner(SUPERVISOR_TAKEOVER).run()
        # Steps before the supervisor action should all pass
        for r in trace.step_results[:-1]:
            self.assertTrue(r.passed, msg=f"step {r.step_index} failed: {r.failure_reason}")


if __name__ == "__main__":
    unittest.main()
