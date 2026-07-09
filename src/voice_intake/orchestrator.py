from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import datetime, timezone
from uuid import uuid4

from .audit import AuditStore, InMemoryAuditStore
from .models import (
    AuditActor,
    AuditEvent,
    AuditEventType,
    CallDisposition,
    CallMode,
    CallSession,
    CallState,
    ConsentType,
    IdentityState,
    ModelProposal,
    SupervisorIntervention,
    SupervisorInterventionType,
    VoiceAction,
)
from .policy import (
    PolicyEngine,
    PolicyProfile,
    baseline_fraud_risk,
    manual_mode_for_consents,
    resolve_consent_capture,
    resolve_entry_state,
)
from .speech_guard import guard_spoken_text
from .validator import ProposalValidator, ValidationOutcome


def _new_id() -> str:
    return str(uuid4())


class VoiceIntakeOrchestrator:
    def __init__(
        self,
        policy_profile: PolicyProfile,
        validator: ProposalValidator,
        audit_store: AuditStore | None = None,
        policy_engine: PolicyEngine | None = None,
    ) -> None:
        self.policy_profile = policy_profile
        self.policy_engine = policy_engine or PolicyEngine(policy_profile)
        self.validator = validator
        self.audit_store = audit_store or InMemoryAuditStore()
        self._reject_counts: dict[tuple[str, str], int] = {}

    def open_session(self, telephony_call_id: str, operator_id: str, attestation) -> CallSession:
        session = CallSession(
            session_id=_new_id(),
            telephony_call_id=telephony_call_id,
            operator_id=operator_id,
            policy_profile_id=self.policy_profile.policy_profile_id,
            template_bundle_version=self.validator.template_bundle.version,
            stir_shaken_attestation=attestation,
        )
        session.metadata["fraud_risk"] = baseline_fraud_risk(attestation)
        self.audit_store.record_audit_event(
            AuditEvent(
                event_id=_new_id(),
                session_id=session.session_id,
                actor=AuditActor.SYSTEM,
                event_type=AuditEventType.SESSION_OPENED,
                proposal_id=None,
                validator_result_id=None,
                affected_fields=[],
                identity_state=IdentityState.UNVERIFIED,
                policy_decision_id=None,
                template_id=None,
                model_version=None,
                service_path="telephony->orchestrator",
                before_after_hash="session_opened",
                details={"fraud_risk": session.metadata["fraud_risk"]},
            )
        )
        return session

    def start_after_opening(self, session: CallSession) -> CallState:
        next_state = resolve_entry_state(self.policy_profile)
        session.current_state = next_state
        self.audit_store.record_audit_event(
            AuditEvent(
                event_id=_new_id(),
                session_id=session.session_id,
                actor=AuditActor.POLICY,
                event_type=AuditEventType.POLICY_PROFILE_PINNED,
                proposal_id=None,
                validator_result_id=None,
                affected_fields=[],
                identity_state=IdentityState.UNVERIFIED,
                policy_decision_id=None,
                template_id=None,
                model_version=None,
                service_path="opening->policy",
                before_after_hash=f"opening->{next_state.value}",
            )
        )
        return next_state

    def capture_consent(
        self, session: CallSession, consent_type: ConsentType, granted: bool
    ) -> CallState:
        prior_artifacts = self.audit_store.consents_for_session(session.session_id)
        artifact, next_state, next_mode = resolve_consent_capture(
            session=session,
            profile=self.policy_profile,
            consent_type=consent_type,
            granted=granted,
            prior_artifacts=prior_artifacts,
        )
        self.audit_store.record_consent_artifact(artifact)
        session.current_state = next_state
        session.mode = next_mode
        self.audit_store.record_audit_event(
            AuditEvent(
                event_id=_new_id(),
                session_id=session.session_id,
                actor=AuditActor.SYSTEM,
                event_type=AuditEventType.CONSENT_CAPTURED,
                proposal_id=None,
                validator_result_id=None,
                affected_fields=["consent"],
                identity_state=IdentityState.UNVERIFIED,
                policy_decision_id=None,
                template_id=None,
                model_version=None,
                service_path="consent->policy",
                before_after_hash=f"{consent_type.value}:{artifact.status}",
                details={"storage_effect": artifact.storage_effect},
            )
        )
        return next_state

    def enter_manual_mode(self, session: CallSession) -> None:
        consents = self.audit_store.consents_for_session(session.session_id)
        profile = manual_mode_for_consents(self.policy_profile, consents)
        if profile.transcription_mode.value == "disabled" and not any(
            "suppress_transcript_persistence" in artifact.storage_effect for artifact in consents
        ):
            raise ValueError(
                "transcription_mode=disabled requires consent suppression recorded in storage_effect"
            )
        session.current_state = CallState.MANUAL_MODE
        session.mode = CallMode.MANUAL
        session.metadata["manual_mode_profile"] = (
            f"{profile.transcription_mode.value}/{profile.transcript_persistence.value}/"
            f"{profile.field_extraction_mode.value}"
        )

    def handle_model_proposal(
        self, session: CallSession, proposal: ModelProposal
    ) -> tuple[ValidationOutcome, VoiceAction | None]:
        proposal.session_id = session.session_id
        self.audit_store.record_proposal(proposal)
        outcome = self.validator.validate(session, proposal)
        self.audit_store.record_validator_result(outcome.validator_result)
        if outcome.policy_decision is not None:
            self.audit_store.record_policy_decision(outcome.policy_decision)
        if not outcome.validator_result.accepted:
            return outcome, self._handle_rejection(session, proposal, outcome)
        # Compute key BEFORE state mutation; otherwise we'd look up against the
        # post-transition state and never clear the counter we actually stored.
        state_key = (session.session_id, session.current_state.value)
        self._reject_counts.pop(state_key, None)
        session.current_state = proposal.requested_transition
        # Natural-language phrasing is spoken only in the demo profile and only
        # when it passes the deterministic speech guard; otherwise the approved
        # template content is rendered exactly as before.
        spoken_text: str | None = None
        spoken_guard = "disabled"
        if self.policy_profile.policy_profile_id == "demo":
            spoken_text, spoken_guard = guard_spoken_text(proposal.spoken_text)
        action = VoiceAction(
            action_type="speak_template",
            template_id=proposal.template_id,
            allowed_variables=proposal.variables,
            interruptible=bool(outcome.template and outcome.template.interruptible),
            timeout_ms=3000,
            spoken_text=spoken_text,
        )
        self.audit_store.record_audit_event(
            AuditEvent(
                event_id=_new_id(),
                session_id=session.session_id,
                actor=AuditActor.MODEL,
                event_type=AuditEventType.VOICE_ACTION_EMITTED,
                proposal_id=proposal.proposal_id,
                validator_result_id=outcome.validator_result.validator_result_id,
                affected_fields=list(proposal.variables),
                identity_state=IdentityState.UNVERIFIED,
                policy_decision_id=(
                    outcome.policy_decision.decision_id if outcome.policy_decision else None
                ),
                template_id=proposal.template_id,
                model_version=proposal.model_version,
                service_path="model->validator->policy->tts",
                before_after_hash=f"accepted->{proposal.requested_transition.value}",
                # The spoken text itself is never persisted - only its hash, so
                # what was spoken can be verified against the chain without
                # storing free-form model output alongside PHI-bearing turns.
                details={
                    "spoken_text_guard": spoken_guard,
                    "spoken_text_sha256": (
                        hashlib.sha256(spoken_text.encode()).hexdigest()
                        if spoken_text
                        else ""
                    ),
                },
            )
        )
        return outcome, action

    def _handle_rejection(
        self, session: CallSession, proposal: ModelProposal, outcome: ValidationOutcome
    ) -> VoiceAction | None:
        # Key by current state (not source_turn_id, which gets a new value per HTTP
        # request and would defeat cross-turn escalation). "Model failed twice in
        # this state" is the intended double-reject semantic.
        key = (session.session_id, session.current_state.value)
        count = self._reject_counts.get(key, 0) + 1
        self._reject_counts[key] = count
        event_type = (
            AuditEventType.VALIDATOR_DOUBLE_REJECT
            if count >= 2
            else AuditEventType.VALIDATOR_REJECTED
        )
        self.audit_store.record_audit_event(
            AuditEvent(
                event_id=_new_id(),
                session_id=session.session_id,
                actor=AuditActor.SYSTEM,
                event_type=event_type,
                proposal_id=proposal.proposal_id,
                validator_result_id=outcome.validator_result.validator_result_id,
                affected_fields=list(proposal.variables),
                identity_state=IdentityState.UNVERIFIED,
                policy_decision_id=(
                    outcome.policy_decision.decision_id if outcome.policy_decision else None
                ),
                template_id=proposal.template_id,
                model_version=proposal.model_version,
                service_path="model->validator",
                before_after_hash=outcome.validator_result.reject_reason.value
                if outcome.validator_result.reject_reason
                else "unknown",
            )
        )
        if count >= 2:
            session.current_state = CallState.HUMAN_TAKEOVER
            session.mode = CallMode.HUMAN_TAKEOVER
        return None

    def handle_model_timeout_grace(self, session: CallSession) -> VoiceAction:
        """Record a model timeout without escalating.

        Emits the hold_message fallback so the caller's next utterance retries
        the LLM. State and mode are unchanged; escalation on repeated timeouts
        stays with handle_model_timeout.
        """
        self.audit_store.record_audit_event(
            AuditEvent(
                event_id=_new_id(),
                session_id=session.session_id,
                actor=AuditActor.SYSTEM,
                event_type=AuditEventType.MODEL_TIMEOUT,
                proposal_id=None,
                validator_result_id=None,
                affected_fields=[],
                identity_state=IdentityState.UNVERIFIED,
                policy_decision_id=None,
                template_id="hold_message",
                model_version=None,
                service_path="orchestrator->tts",
                before_after_hash="model_timeout_grace",
                details={"grace": True},
            )
        )
        return VoiceAction(
            action_type="speak_template",
            template_id="hold_message",
            allowed_variables={},
            interruptible=True,
            timeout_ms=3000,
        )

    def handle_model_timeout(self, session: CallSession, buffered_audio_ref: str | None) -> None:
        session.current_state = CallState.HUMAN_TAKEOVER
        session.mode = CallMode.HUMAN_TAKEOVER
        if buffered_audio_ref:
            session.metadata["buffered_audio_ref"] = buffered_audio_ref
        self.audit_store.record_audit_event(
            AuditEvent(
                event_id=_new_id(),
                session_id=session.session_id,
                actor=AuditActor.SYSTEM,
                event_type=AuditEventType.MODEL_TIMEOUT,
                proposal_id=None,
                validator_result_id=None,
                affected_fields=[],
                identity_state=IdentityState.UNVERIFIED,
                policy_decision_id=None,
                template_id="hold_message",
                model_version=None,
                service_path="orchestrator->tts->human_takeover",
                before_after_hash="model_timeout",
                details={"buffered_audio_ref": buffered_audio_ref or ""},
            )
        )

    def abandon_call(self, session: CallSession) -> None:
        if session.current_state == CallState.EMERGENCY_EXIT:
            session.disposition = CallDisposition.EMERGENCY_ABANDONED
            event_type = AuditEventType.EMERGENCY_ABANDONED
        else:
            session.disposition = CallDisposition.CALLER_ABANDONED
            event_type = AuditEventType.CALL_ABANDONED
        session.ended_at = datetime.now(timezone.utc)
        self.audit_store.record_audit_event(
            AuditEvent(
                event_id=_new_id(),
                session_id=session.session_id,
                actor=AuditActor.SYSTEM,
                event_type=event_type,
                proposal_id=None,
                validator_result_id=None,
                affected_fields=[],
                identity_state=IdentityState.UNVERIFIED,
                policy_decision_id=None,
                template_id=None,
                model_version=None,
                service_path="call_end",
                before_after_hash=session.disposition.value,
            )
        )

    def record_supervisor_intervention(
        self,
        session: CallSession,
        intervention_type: SupervisorInterventionType,
        reason_code: str,
        notes: str,
    ) -> None:
        intervention = SupervisorIntervention(
            session_id=session.session_id,
            intervention_type=intervention_type,
            reason_code=reason_code,
            alert_ts=datetime.now(timezone.utc),
            ack_ts=datetime.now(timezone.utc),
            takeover_ts=(
                datetime.now(timezone.utc)
                if intervention_type == SupervisorInterventionType.FORCED_TAKEOVER
                else None
            ),
            notes=notes,
        )
        self.audit_store.record_supervisor_intervention(intervention)
        if intervention_type == SupervisorInterventionType.FORCED_TAKEOVER:
            session.current_state = CallState.HUMAN_TAKEOVER
            session.mode = CallMode.HUMAN_TAKEOVER
