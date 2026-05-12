from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CallState(StrEnum):
    OPENING = "opening"
    CONSENT_RECORDING = "consent_recording"
    CONSENT_AI_ASSISTANCE = "consent_ai_assistance"
    CONSENT_DECLINED = "consent_declined"
    IDENTITY_CAPTURE = "identity_capture"
    DEMOGRAPHICS = "demographics"
    INSURANCE = "insurance"
    REASON_FOR_VISIT = "reason_for_visit"
    READBACK = "readback"
    DISPOSITION = "disposition"
    CLOSE = "close"
    HUMAN_TAKEOVER = "human_takeover"
    EMERGENCY_EXIT = "emergency_exit"
    MANUAL_MODE = "manual_mode"


class CallMode(StrEnum):
    AI_LED = "ai_led"
    MANUAL = "manual"
    HUMAN_TAKEOVER = "human_takeover"
    EMERGENCY = "emergency"


class StirShakenAttestation(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    UNKNOWN = "unknown"


class CallDisposition(StrEnum):
    INTAKE_COMPLETE = "intake_complete"
    PARTIAL_FOLLOWUP = "partial_followup"
    TRANSFERRED_TO_HUMAN = "transferred_to_human"
    EMERGENCY_REDIRECT = "emergency_redirect"
    CALLER_ABANDONED = "caller_abandoned"
    EMERGENCY_ABANDONED = "emergency_abandoned"


class Speaker(StrEnum):
    AI = "ai"
    CALLER = "caller"
    SUPERVISOR = "supervisor"


class VerificationStatus(StrEnum):
    UNVERIFIED = "unverified"
    PENDING_CONFIRMATION = "pending_confirmation"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    HUMAN_RESOLVED = "human_resolved"


class ConsentType(StrEnum):
    RECORDING = "recording_consent"
    AI_ASSISTANCE = "ai_assistance_consent"


class RejectReason(StrEnum):
    INVALID_TEMPLATE_ID = "invalid_template_id"
    DISALLOWED_TRANSITION = "disallowed_transition"
    VARIABLE_SCHEMA_VIOLATION = "variable_schema_violation"
    OUT_OF_VOCABULARY_PHI = "out_of_vocabulary_phi"
    POLICY_DENIED = "policy_denied"


class IdentityState(StrEnum):
    UNVERIFIED = "unverified"
    INTAKE_THRESHOLD_MET = "intake_threshold_met"
    THRESHOLD_FAILED = "threshold_failed"
    HUMAN_ASSERTED = "human_asserted"


class SupervisorInterventionType(StrEnum):
    FIELD_EDIT = "field_edit"
    STATUS_OVERRIDE = "status_override"
    FORCED_TAKEOVER = "forced_takeover"
    EMERGENCY_OVERRIDE = "emergency_override"
    CALL_TERMINATE = "call_terminate"


class TranscriptionMode(StrEnum):
    ACTIVE = "active"
    PASSIVE_ONLY = "passive_only"
    DISABLED = "disabled"


class TranscriptPersistence(StrEnum):
    STANDARD = "standard"
    SUPPRESSED = "suppressed"
    AUDIT_ONLY = "audit_only"


class FieldExtractionMode(StrEnum):
    ACTIVE = "active"
    SUPERVISOR_CONFIRMED_ONLY = "supervisor_confirmed_only"
    DISABLED = "disabled"


class AuditActor(StrEnum):
    SYSTEM = "system"
    MODEL = "model"
    POLICY = "policy_engine"
    SUPERVISOR = "supervisor"


class AuditEventType(StrEnum):
    PROPOSAL_ACCEPTED = "proposal_accepted"
    VALIDATOR_REJECTED = "validator_rejected"
    VALIDATOR_DOUBLE_REJECT = "validator_double_reject"
    MODEL_TIMEOUT = "model_timeout"
    EMERGENCY_ABANDONED = "emergency_abandoned"
    CALL_ABANDONED = "call_abandoned"
    CONSENT_CAPTURED = "consent_captured"
    SESSION_OPENED = "session_opened"
    VOICE_ACTION_EMITTED = "voice_action_emitted"
    POLICY_PROFILE_PINNED = "policy_profile_pinned"


@dataclass(slots=True)
class CallSession:
    session_id: str
    telephony_call_id: str
    operator_id: str
    current_state: CallState = CallState.OPENING
    mode: CallMode = CallMode.AI_LED
    policy_profile_id: str = "default"
    template_bundle_version: str = "default"
    language_status: str = "en_supported"
    stir_shaken_attestation: StirShakenAttestation = StirShakenAttestation.UNKNOWN
    started_at: datetime = field(default_factory=utc_now)
    ended_at: datetime | None = None
    disposition: CallDisposition | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class VoiceTurn:
    turn_id: str
    speaker: Speaker
    audio_ref: str | None
    retention_policy_id: str
    transcript: str
    partial_or_final: str
    asr_confidence: float
    barge_in: bool
    start_ts: datetime
    end_ts: datetime


@dataclass(slots=True)
class NormalizedSegment:
    field_type: str
    raw_span: str
    normalized_value: str
    normalization_confidence: float


@dataclass(slots=True)
class NormalizationResult:
    turn_id: str
    raw_transcript: str
    normalized_segments: list[NormalizedSegment]
    created_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class FieldCandidate:
    field_name: str
    candidate_value: str
    confidence: float
    verification_status: VerificationStatus
    source_turn_ids: list[str]
    sensitivity_class: str


@dataclass(slots=True)
class ConsentArtifact:
    session_id: str
    consent_type: ConsentType
    status: str
    jurisdiction_policy_id: str
    captured_at: datetime
    capture_mode: str
    storage_effect: str


@dataclass(slots=True)
class ModelProposal:
    proposal_id: str
    source_turn_id: str
    template_id: str
    variables: dict[str, str]
    requested_transition: CallState
    model_version: str
    session_id: str | None = None


@dataclass(slots=True)
class ValidatorResult:
    validator_result_id: str
    proposal_id: str
    accepted: bool
    reject_reason: RejectReason | None
    created_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class PolicyDecision:
    decision_id: str
    session_id: str
    state: CallState
    requested_action: str
    allowed: bool
    reason_code: str
    policy_profile_id: str
    template_id: str
    created_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class ManualModeProfile:
    tts_enabled: bool = False
    policy_engine_mode: str = "observe_only"
    transcription_mode: TranscriptionMode = TranscriptionMode.ACTIVE
    transcript_persistence: TranscriptPersistence = TranscriptPersistence.AUDIT_ONLY
    field_extraction_mode: FieldExtractionMode = FieldExtractionMode.SUPERVISOR_CONFIRMED_ONLY
    audit_mode: str = "full"
    supervisor_notification_mode: str = "active_alert"

    def __post_init__(self) -> None:
        if isinstance(self.transcription_mode, str):
            self.transcription_mode = TranscriptionMode(self.transcription_mode)
        if isinstance(self.transcript_persistence, str):
            self.transcript_persistence = TranscriptPersistence(self.transcript_persistence)
        if isinstance(self.field_extraction_mode, str):
            self.field_extraction_mode = FieldExtractionMode(self.field_extraction_mode)
        if self.field_extraction_mode == FieldExtractionMode.DISABLED:
            raise ValueError("field_extraction_mode=disabled is forbidden in manual_mode")


@dataclass(slots=True)
class AuditEvent:
    event_id: str
    session_id: str
    actor: AuditActor
    event_type: AuditEventType | str
    proposal_id: str | None
    validator_result_id: str | None
    affected_fields: list[str]
    identity_state: IdentityState
    policy_decision_id: str | None
    template_id: str | None
    model_version: str | None
    service_path: str
    before_after_hash: str
    timestamp: datetime = field(default_factory=utc_now)
    details: dict[str, str] = field(default_factory=dict)
    chain_hash: str = ""
    chain_index: int | None = None


@dataclass(slots=True)
class SupervisorIntervention:
    session_id: str
    intervention_type: SupervisorInterventionType
    reason_code: str
    alert_ts: datetime
    ack_ts: datetime | None
    takeover_ts: datetime | None
    notes: str


@dataclass(slots=True)
class VoiceAction:
    action_type: str
    template_id: str
    allowed_variables: dict[str, str]
    interruptible: bool
    timeout_ms: int


# ── Appointment scheduling (demo flow) ────────────────────────────────────────

class AppointmentStatus(StrEnum):
    COLLECTING = "collecting"
    SCHEDULED = "scheduled"
    STAFF_REVIEW = "staff_review"
    EMERGENCY_REDIRECT = "emergency_redirect"
    HUMAN_TAKEOVER = "human_takeover"


class InsuranceNetworkStatus(StrEnum):
    IN_NETWORK = "in_network"
    OUT_OF_NETWORK = "out_of_network"
    UNKNOWN = "unknown"
    NOT_PROVIDED = "not_provided"


@dataclass(slots=True)
class AppointmentRequest:
    appointment_id: str                   # server-generated (uuid4); never from frontend
    session_id: str
    reason_for_visit: str | None = None
    patient_name: str | None = None
    date_of_birth: str | None = None      # ISO YYYY-MM-DD
    insurance_name: str | None = None
    insurance_network_status: InsuranceNetworkStatus = InsuranceNetworkStatus.UNKNOWN
    appointment_status: AppointmentStatus = AppointmentStatus.COLLECTING
    scheduled_slot: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
