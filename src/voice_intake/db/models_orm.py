from __future__ import annotations

import json
from datetime import datetime
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from voice_intake.models import (
    AppointmentRequest,
    AppointmentStatus,
    AuditActor,
    AuditEvent,
    AuditEventType,
    CallDisposition,
    CallMode,
    CallSession,
    CallState,
    ConsentArtifact,
    ConsentType,
    IdentityState,
    InsuranceNetworkStatus,
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

from .base import Base


def _uid() -> str:
    return str(uuid4())


class SessionORM(Base):
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String, primary_key=True)
    telephony_call_id: Mapped[str] = mapped_column(String, nullable=False)
    operator_id: Mapped[str] = mapped_column(String, nullable=False)
    current_state: Mapped[str] = mapped_column(String, nullable=False)
    mode: Mapped[str] = mapped_column(String, nullable=False)
    policy_profile_id: Mapped[str] = mapped_column(String, nullable=False)
    template_bundle_version: Mapped[str] = mapped_column(String, nullable=False)
    language_status: Mapped[str] = mapped_column(String, nullable=False)
    stir_shaken_attestation: Mapped[str] = mapped_column(String, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    disposition: Mapped[str | None] = mapped_column(String, nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    # Serialized _reject_counts dict for Phase 3 cross-request continuity
    reject_counts_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    # Caller ANI for patient lookup and prior call history (Phase 4)
    caller_phone_normalized: Mapped[str | None] = mapped_column(String, nullable=True, index=True)

    @classmethod
    def from_domain(
        cls,
        session: CallSession,
        reject_counts: dict | None = None,
        caller_phone: str | None = None,
    ) -> "SessionORM":
        return cls(
            session_id=session.session_id,
            telephony_call_id=session.telephony_call_id,
            operator_id=session.operator_id,
            current_state=session.current_state.value,
            mode=session.mode.value,
            policy_profile_id=session.policy_profile_id,
            template_bundle_version=session.template_bundle_version,
            language_status=session.language_status,
            stir_shaken_attestation=session.stir_shaken_attestation.value,
            started_at=session.started_at,
            ended_at=session.ended_at,
            disposition=session.disposition.value if session.disposition else None,
            metadata_json=json.dumps(session.metadata),
            reject_counts_json=json.dumps(
                {f"{k[0]}|{k[1]}": v for k, v in (reject_counts or {}).items()}
            ),
            caller_phone_normalized=caller_phone,
        )

    def to_domain(self) -> CallSession:
        session = CallSession(
            session_id=self.session_id,
            telephony_call_id=self.telephony_call_id,
            operator_id=self.operator_id,
            current_state=CallState(self.current_state),
            mode=CallMode(self.mode),
            policy_profile_id=self.policy_profile_id,
            template_bundle_version=self.template_bundle_version,
            language_status=self.language_status,
            stir_shaken_attestation=StirShakenAttestation(self.stir_shaken_attestation),
            started_at=self.started_at,
            ended_at=self.ended_at,
            disposition=CallDisposition(self.disposition) if self.disposition else None,
            metadata=json.loads(self.metadata_json),
        )
        return session

    def reject_counts(self) -> dict[tuple[str, str], int]:
        raw: dict[str, int] = json.loads(self.reject_counts_json)
        return {tuple(k.split("|", 1)): v for k, v in raw.items()}  # type: ignore[return-value]

    def set_reject_counts(self, counts: dict[tuple[str, str], int]) -> None:
        self.reject_counts_json = json.dumps(
            {f"{k[0]}|{k[1]}": v for k, v in counts.items()}
        )


class VoiceTurnORM(Base):
    __tablename__ = "voice_turns"

    turn_id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    speaker: Mapped[str] = mapped_column(String, nullable=False)
    audio_ref: Mapped[str | None] = mapped_column(String, nullable=True)
    retention_policy_id: Mapped[str] = mapped_column(String, nullable=False)
    transcript: Mapped[str] = mapped_column(Text, nullable=False)
    partial_or_final: Mapped[str] = mapped_column(String, nullable=False)
    asr_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    barge_in: Mapped[int] = mapped_column(Integer, nullable=False)
    start_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    @classmethod
    def from_domain(cls, session_id: str, turn: VoiceTurn) -> "VoiceTurnORM":
        return cls(
            turn_id=turn.turn_id,
            session_id=session_id,
            speaker=turn.speaker.value,
            audio_ref=turn.audio_ref,
            retention_policy_id=turn.retention_policy_id,
            transcript=turn.transcript,
            partial_or_final=turn.partial_or_final,
            asr_confidence=turn.asr_confidence,
            barge_in=int(turn.barge_in),
            start_ts=turn.start_ts,
            end_ts=turn.end_ts,
        )

    def to_domain(self) -> VoiceTurn:
        return VoiceTurn(
            turn_id=self.turn_id,
            speaker=Speaker(self.speaker),
            audio_ref=self.audio_ref,
            retention_policy_id=self.retention_policy_id,
            transcript=self.transcript,
            partial_or_final=self.partial_or_final,
            asr_confidence=self.asr_confidence,
            barge_in=bool(self.barge_in),
            start_ts=self.start_ts,
            end_ts=self.end_ts,
        )


class AuditEventORM(Base):
    __tablename__ = "audit_events"
    __table_args__ = (
        UniqueConstraint("session_id", "chain_index", name="uq_audit_events_session_chain"),
    )

    event_id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    actor: Mapped[str] = mapped_column(String, nullable=False)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    proposal_id: Mapped[str | None] = mapped_column(String, nullable=True)
    validator_result_id: Mapped[str | None] = mapped_column(String, nullable=True)
    affected_fields_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    identity_state: Mapped[str] = mapped_column(String, nullable=False)
    policy_decision_id: Mapped[str | None] = mapped_column(String, nullable=True)
    template_id: Mapped[str | None] = mapped_column(String, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String, nullable=True)
    service_path: Mapped[str] = mapped_column(String, nullable=False)
    before_after_hash: Mapped[str] = mapped_column(String, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    details_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    chain_hash: Mapped[str] = mapped_column(String, nullable=False, default="")
    chain_index: Mapped[int | None] = mapped_column(Integer, nullable=True)

    @classmethod
    def from_domain(cls, event: AuditEvent) -> "AuditEventORM":
        event_type = (
            event.event_type.value
            if isinstance(event.event_type, AuditEventType)
            else str(event.event_type)
        )
        return cls(
            event_id=event.event_id,
            session_id=event.session_id,
            actor=event.actor.value,
            event_type=event_type,
            proposal_id=event.proposal_id,
            validator_result_id=event.validator_result_id,
            affected_fields_json=json.dumps(event.affected_fields),
            identity_state=event.identity_state.value,
            policy_decision_id=event.policy_decision_id,
            template_id=event.template_id,
            model_version=event.model_version,
            service_path=event.service_path,
            before_after_hash=event.before_after_hash,
            timestamp=event.timestamp,
            details_json=json.dumps(event.details),
            chain_hash=event.chain_hash,
            chain_index=event.chain_index,
        )

    def to_domain(self) -> AuditEvent:
        try:
            event_type: AuditEventType | str = AuditEventType(self.event_type)
        except ValueError:
            event_type = self.event_type
        return AuditEvent(
            event_id=self.event_id,
            session_id=self.session_id,
            actor=AuditActor(self.actor),
            event_type=event_type,
            proposal_id=self.proposal_id,
            validator_result_id=self.validator_result_id,
            affected_fields=json.loads(self.affected_fields_json),
            identity_state=IdentityState(self.identity_state),
            policy_decision_id=self.policy_decision_id,
            template_id=self.template_id,
            model_version=self.model_version,
            service_path=self.service_path,
            before_after_hash=self.before_after_hash,
            timestamp=self.timestamp,
            details=json.loads(self.details_json),
            chain_hash=self.chain_hash,
            chain_index=self.chain_index,
        )


class ModelProposalORM(Base):
    __tablename__ = "model_proposals"

    proposal_id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    source_turn_id: Mapped[str] = mapped_column(String, nullable=False)
    template_id: Mapped[str] = mapped_column(String, nullable=False)
    variables_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    requested_transition: Mapped[str] = mapped_column(String, nullable=False)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    @classmethod
    def from_domain(cls, session_id: str, proposal: ModelProposal) -> "ModelProposalORM":
        return cls(
            proposal_id=proposal.proposal_id,
            session_id=session_id,
            source_turn_id=proposal.source_turn_id,
            template_id=proposal.template_id,
            variables_json=json.dumps(proposal.variables),
            requested_transition=proposal.requested_transition.value,
            model_version=proposal.model_version,
        )

    def to_domain(self) -> ModelProposal:
        return ModelProposal(
            proposal_id=self.proposal_id,
            source_turn_id=self.source_turn_id,
            template_id=self.template_id,
            variables=json.loads(self.variables_json),
            requested_transition=CallState(self.requested_transition),
            model_version=self.model_version,
        )


class ValidatorResultORM(Base):
    __tablename__ = "validator_results"

    validator_result_id: Mapped[str] = mapped_column(String, primary_key=True)
    proposal_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    accepted: Mapped[int] = mapped_column(Integer, nullable=False)
    reject_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    @classmethod
    def from_domain(cls, result: ValidatorResult) -> "ValidatorResultORM":
        return cls(
            validator_result_id=result.validator_result_id,
            proposal_id=result.proposal_id,
            accepted=int(result.accepted),
            reject_reason=result.reject_reason.value if result.reject_reason else None,
        )

    def to_domain(self) -> ValidatorResult:
        return ValidatorResult(
            validator_result_id=self.validator_result_id,
            proposal_id=self.proposal_id,
            accepted=bool(self.accepted),
            reject_reason=RejectReason(self.reject_reason) if self.reject_reason else None,
            created_at=self.created_at,
        )


class ConsentArtifactORM(Base):
    __tablename__ = "consent_artifacts"

    artifact_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uid)
    session_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    consent_type: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    jurisdiction_policy_id: Mapped[str] = mapped_column(String, nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    capture_mode: Mapped[str] = mapped_column(String, nullable=False)
    storage_effect: Mapped[str] = mapped_column(String, nullable=False)

    @classmethod
    def from_domain(cls, artifact: ConsentArtifact) -> "ConsentArtifactORM":
        return cls(
            artifact_id=_uid(),
            session_id=artifact.session_id,
            consent_type=artifact.consent_type.value,
            status=artifact.status,
            jurisdiction_policy_id=artifact.jurisdiction_policy_id,
            captured_at=artifact.captured_at,
            capture_mode=artifact.capture_mode,
            storage_effect=artifact.storage_effect,
        )

    def to_domain(self) -> ConsentArtifact:
        return ConsentArtifact(
            session_id=self.session_id,
            consent_type=ConsentType(self.consent_type),
            status=self.status,
            jurisdiction_policy_id=self.jurisdiction_policy_id,
            captured_at=self.captured_at,
            capture_mode=self.capture_mode,
            storage_effect=self.storage_effect,
        )


class SupervisorInterventionORM(Base):
    __tablename__ = "supervisor_interventions"

    intervention_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uid)
    session_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    intervention_type: Mapped[str] = mapped_column(String, nullable=False)
    reason_code: Mapped[str] = mapped_column(String, nullable=False)
    notes: Mapped[str] = mapped_column(Text, nullable=False)
    alert_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ack_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    takeover_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    @classmethod
    def from_domain(cls, intervention: SupervisorIntervention) -> "SupervisorInterventionORM":
        return cls(
            intervention_id=_uid(),
            session_id=intervention.session_id,
            intervention_type=intervention.intervention_type.value,
            reason_code=intervention.reason_code,
            notes=intervention.notes,
            alert_ts=intervention.alert_ts,
            ack_ts=intervention.ack_ts,
            takeover_ts=intervention.takeover_ts,
        )

    def to_domain(self) -> SupervisorIntervention:
        return SupervisorIntervention(
            session_id=self.session_id,
            intervention_type=SupervisorInterventionType(self.intervention_type),
            reason_code=self.reason_code,
            notes=self.notes,
            alert_ts=self.alert_ts,
            ack_ts=self.ack_ts,
            takeover_ts=self.takeover_ts,
        )


class AppointmentRequestORM(Base):
    """One row per session; upserted by DemoRouter after proposal validation."""

    __tablename__ = "appointment_requests"
    __table_args__ = (
        UniqueConstraint("session_id", name="uq_appointment_requests_session"),
    )

    appointment_id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    reason_for_visit: Mapped[str | None] = mapped_column(Text, nullable=True)
    patient_name: Mapped[str | None] = mapped_column(String, nullable=True)
    date_of_birth: Mapped[str | None] = mapped_column(String, nullable=True)
    insurance_name: Mapped[str | None] = mapped_column(String, nullable=True)
    insurance_network_status: Mapped[str] = mapped_column(
        String, nullable=False, default=InsuranceNetworkStatus.UNKNOWN.value
    )
    appointment_status: Mapped[str] = mapped_column(
        String, nullable=False, default=AppointmentStatus.COLLECTING.value
    )
    scheduled_slot: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    @classmethod
    def from_domain(cls, req: AppointmentRequest) -> "AppointmentRequestORM":
        return cls(
            appointment_id=req.appointment_id,
            session_id=req.session_id,
            reason_for_visit=req.reason_for_visit,
            patient_name=req.patient_name,
            date_of_birth=req.date_of_birth,
            insurance_name=req.insurance_name,
            insurance_network_status=req.insurance_network_status.value,
            appointment_status=req.appointment_status.value,
            scheduled_slot=req.scheduled_slot,
            created_at=req.created_at,
            updated_at=req.updated_at,
        )

    def to_domain(self) -> AppointmentRequest:
        return AppointmentRequest(
            appointment_id=self.appointment_id,
            session_id=self.session_id,
            reason_for_visit=self.reason_for_visit,
            patient_name=self.patient_name,
            date_of_birth=self.date_of_birth,
            insurance_name=self.insurance_name,
            insurance_network_status=InsuranceNetworkStatus(self.insurance_network_status),
            appointment_status=AppointmentStatus(self.appointment_status),
            scheduled_slot=self.scheduled_slot,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


class PolicyDecisionORM(Base):
    __tablename__ = "policy_decisions"

    decision_id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    state: Mapped[str] = mapped_column(String, nullable=False)
    requested_action: Mapped[str] = mapped_column(String, nullable=False)
    allowed: Mapped[int] = mapped_column(Integer, nullable=False)
    reason_code: Mapped[str] = mapped_column(String, nullable=False)
    policy_profile_id: Mapped[str] = mapped_column(String, nullable=False)
    template_id: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    @classmethod
    def from_domain(cls, decision: PolicyDecision) -> "PolicyDecisionORM":
        return cls(
            decision_id=decision.decision_id,
            session_id=decision.session_id,
            state=decision.state.value,
            requested_action=decision.requested_action,
            allowed=int(decision.allowed),
            reason_code=decision.reason_code,
            policy_profile_id=decision.policy_profile_id,
            template_id=decision.template_id,
            created_at=decision.created_at,
        )

    def to_domain(self) -> PolicyDecision:
        return PolicyDecision(
            decision_id=self.decision_id,
            session_id=self.session_id,
            state=CallState(self.state),
            requested_action=self.requested_action,
            allowed=bool(self.allowed),
            reason_code=self.reason_code,
            policy_profile_id=self.policy_profile_id,
            template_id=self.template_id,
            created_at=self.created_at,
        )
