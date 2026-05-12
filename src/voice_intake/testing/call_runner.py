from __future__ import annotations

import json
from dataclasses import dataclass, field
from uuid import uuid4

from voice_intake.audit import InMemoryAuditStore
from voice_intake.models import (
    CallDisposition,
    CallSession,
    CallState,
    ConsentType,
    StirShakenAttestation,
    SupervisorInterventionType,
    VoiceAction,
)
from voice_intake.normalization import extract_field_candidates, normalize_turn
from voice_intake.orchestrator import VoiceIntakeOrchestrator
from voice_intake.policy import PolicyEngine, PolicyProfile
from voice_intake.prompting import build_model_prompt
from voice_intake.templates import DEFAULT_TEMPLATE_BUNDLE
from voice_intake.validator import ProposalValidator, ValidationOutcome

from .mock_asr import MockASR
from .mock_llm import MockLLM


def _new_id() -> str:
    return str(uuid4())


@dataclass
class ConsentGrant:
    consent_type: ConsentType
    granted: bool


@dataclass
class SupervisorAction:
    intervention_type: SupervisorInterventionType
    reason_code: str
    notes: str = ""


@dataclass
class ScriptStep:
    caller_text: str
    expected_state_after: CallState
    # Exactly one of the following should be set per step
    llm_response: dict | None = None        # {template_id, variables, requested_transition}
    consent_grant: ConsentGrant | None = None
    supervisor_action: SupervisorAction | None = None
    # For LLM steps
    expected_action_template: str | None = None
    expect_rejection: bool = False
    # Force a specific turn_id (use to share a turn across two proposals for double-reject)
    turn_id: str | None = None


@dataclass
class CallScript:
    label: str
    attestation: StirShakenAttestation
    policy_profile: PolicyProfile
    steps: list[ScriptStep]
    expected_final_disposition: CallDisposition | None = None


@dataclass
class StepResult:
    step_index: int
    caller_text: str
    emitted_action: VoiceAction | None
    actual_state: CallState
    expected_state: CallState
    validation_outcome: ValidationOutcome | None
    passed: bool
    failure_reason: str | None


@dataclass
class CallTrace:
    session: CallSession
    step_results: list[StepResult]
    all_passed: bool
    audit_event_count: int
    llm_prompt_snapshots: list[dict] = field(default_factory=list)


class ScriptedCallRunner:
    def __init__(self, script: CallScript) -> None:
        self.script = script
        self.audit_store = InMemoryAuditStore()
        policy_engine = PolicyEngine(script.policy_profile)
        validator = ProposalValidator(DEFAULT_TEMPLATE_BUNDLE, policy_engine)
        self.orchestrator = VoiceIntakeOrchestrator(
            policy_profile=script.policy_profile,
            validator=validator,
            audit_store=self.audit_store,
            policy_engine=policy_engine,
        )
        self.mock_asr = MockASR()
        # Build the MockLLM queue from all LLM steps in the script
        llm_script = [
            step.llm_response
            for step in script.steps
            if step.llm_response is not None
        ]
        self.mock_llm = MockLLM(llm_script)
        self._prompt_snapshots: list[dict] = []

    def run(self) -> CallTrace:
        session = self.orchestrator.open_session(
            telephony_call_id="mock-call-001",
            operator_id="mock-operator",
            attestation=self.script.attestation,
        )
        self.orchestrator.start_after_opening(session)

        step_results: list[StepResult] = []
        for i, step in enumerate(self.script.steps):
            result = self._run_step(i, step, session)
            step_results.append(result)

        all_passed = all(r.passed for r in step_results)
        return CallTrace(
            session=session,
            step_results=step_results,
            all_passed=all_passed,
            audit_event_count=len(self.audit_store.audit_events),
            llm_prompt_snapshots=list(self._prompt_snapshots),
        )

    def _run_step(self, index: int, step: ScriptStep, session: CallSession) -> StepResult:
        if step.consent_grant is not None:
            return self._run_consent_step(index, step, session)
        if step.supervisor_action is not None:
            return self._run_supervisor_step(index, step, session)
        return self._run_llm_step(index, step, session)

    def _run_consent_step(
        self, index: int, step: ScriptStep, session: CallSession
    ) -> StepResult:
        cg = step.consent_grant
        assert cg is not None
        self.orchestrator.capture_consent(session, cg.consent_type, cg.granted)
        passed = session.current_state == step.expected_state_after
        return StepResult(
            step_index=index,
            caller_text=step.caller_text,
            emitted_action=None,
            actual_state=session.current_state,
            expected_state=step.expected_state_after,
            validation_outcome=None,
            passed=passed,
            failure_reason=(
                None
                if passed
                else f"consent: expected {step.expected_state_after.value}, got {session.current_state.value}"
            ),
        )

    def _run_supervisor_step(
        self, index: int, step: ScriptStep, session: CallSession
    ) -> StepResult:
        sa = step.supervisor_action
        assert sa is not None
        self.orchestrator.record_supervisor_intervention(
            session, sa.intervention_type, sa.reason_code, sa.notes
        )
        passed = session.current_state == step.expected_state_after
        return StepResult(
            step_index=index,
            caller_text=step.caller_text,
            emitted_action=None,
            actual_state=session.current_state,
            expected_state=step.expected_state_after,
            validation_outcome=None,
            passed=passed,
            failure_reason=(
                None
                if passed
                else f"supervisor: expected {step.expected_state_after.value}, got {session.current_state.value}"
            ),
        )

    def _run_llm_step(
        self, index: int, step: ScriptStep, session: CallSession
    ) -> StepResult:
        turn = self.mock_asr.transcribe(step.caller_text, turn_id=step.turn_id)
        norm = normalize_turn(turn)
        candidates = extract_field_candidates(norm)
        available_templates = list(
            self.orchestrator.validator.template_bundle.templates.values()
        )
        prompt = build_model_prompt(session, [turn], candidates, available_templates)
        self._prompt_snapshots.append(prompt)

        proposal = self.mock_llm.propose(prompt, turn.turn_id)
        outcome, action = self.orchestrator.handle_model_proposal(session, proposal)

        if step.expected_action_template is not None:
            action_ok = action is not None and action.template_id == step.expected_action_template
        elif step.expect_rejection:
            action_ok = action is None
        else:
            action_ok = action is not None

        state_ok = session.current_state == step.expected_state_after
        passed = state_ok and action_ok

        reasons: list[str] = []
        if not state_ok:
            reasons.append(
                f"state: expected {step.expected_state_after.value}, got {session.current_state.value}"
            )
        if not action_ok:
            got_template = action.template_id if action else "None"
            reasons.append(
                f"action: expected {step.expected_action_template or ('None' if step.expect_rejection else 'any')}, got {got_template}"
            )

        return StepResult(
            step_index=index,
            caller_text=step.caller_text,
            emitted_action=action,
            actual_state=session.current_state,
            expected_state=step.expected_state_after,
            validation_outcome=outcome,
            passed=passed,
            failure_reason="; ".join(reasons) if reasons else None,
        )

    def prompt_contains_raw_phi(self) -> list[str]:
        phi_patterns = [r"\d{10}", r"\d{3}[-.\s]\d{3}[-.\s]\d{4}"]
        import re
        hits: list[str] = []
        for snapshot in self._prompt_snapshots:
            text = json.dumps(snapshot)
            for pattern in phi_patterns:
                matches = re.findall(pattern, text)
                hits.extend(matches)
        return hits
