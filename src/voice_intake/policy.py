from __future__ import annotations

from dataclasses import dataclass, field, replace
from uuid import uuid4

from .models import (
    CallMode,
    CallSession,
    CallState,
    ConsentArtifact,
    ConsentType,
    ManualModeProfile,
    PolicyDecision,
    StirShakenAttestation,
    TranscriptPersistence,
    TranscriptionMode,
)
from .state_machine import allowed_targets


def _new_id() -> str:
    return str(uuid4())


@dataclass(slots=True)
class PolicyProfile:
    policy_profile_id: str = "default"
    require_recording_consent: bool = True
    require_ai_assistance_consent: bool = True
    allow_ai_without_recording: bool = True
    allow_service_after_recording_decline: bool = True
    allow_service_after_ai_decline: bool = True
    allow_service_after_both_declined: bool = True
    manual_mode_profile: ManualModeProfile = field(default_factory=ManualModeProfile)
    # Override the fallback entry state when both consent flags are False.
    # Consent flow ignores this (the consent states are always forced first when
    # require_recording_consent / require_ai_assistance_consent are True).
    default_entry_state: CallState = CallState.IDENTITY_CAPTURE


def baseline_fraud_risk(attestation: StirShakenAttestation) -> str:
    if attestation == StirShakenAttestation.A:
        return "standard"
    if attestation == StirShakenAttestation.B:
        return "elevated"
    return "high"


def resolve_entry_state(profile: PolicyProfile) -> CallState:
    if profile.require_recording_consent:
        return CallState.CONSENT_RECORDING
    if profile.require_ai_assistance_consent:
        return CallState.CONSENT_AI_ASSISTANCE
    return profile.default_entry_state


_DEMO_POLICY_HIDDEN_TRANSITIONS = frozenset(
    {
        CallState.CONSENT_RECORDING,
        CallState.CONSENT_AI_ASSISTANCE,
        CallState.CONSENT_DECLINED,
        CallState.MANUAL_MODE,
    }
)


def model_allowed_transitions(
    profile: PolicyProfile, state: CallState
) -> frozenset[CallState]:
    """Return state-machine targets filtered for what the model should see.

    Raw state-machine transitions remain authoritative for the validator.
    This helper hides policy-disallowed targets so the LLM doesn't propose
    them - e.g. demo policy hides consent and manual-mode targets because
    the public scheduling demo is not actually recording calls.
    """
    targets = set(allowed_targets(state))
    if profile.policy_profile_id == "demo":
        targets.difference_update(_DEMO_POLICY_HIDDEN_TRANSITIONS)
    return frozenset(targets)


def demo_policy_profile() -> PolicyProfile:
    """Policy profile for the public scheduling demo.

    Skips recording and AI-assistance consent gates entirely and enters the
    call at OPENING so DemoRouter can greet before collecting fields.
    """
    return PolicyProfile(
        policy_profile_id="demo",
        require_recording_consent=False,
        require_ai_assistance_consent=False,
        default_entry_state=CallState.OPENING,
    )


def _recording_storage_effect(granted: bool) -> str:
    return "standard_retention" if granted else "suppress_audio_recording;suppress_transcript_persistence"


def _ai_storage_effect(granted: bool) -> str:
    return "standard_retention" if granted else "disable_ai_led_flow"


def manual_mode_for_consents(
    profile: PolicyProfile, consent_artifacts: list[ConsentArtifact]
) -> ManualModeProfile:
    declined_recording = any(
        artifact.consent_type == ConsentType.RECORDING and artifact.status == "declined"
        for artifact in consent_artifacts
    )
    declined_ai = any(
        artifact.consent_type == ConsentType.AI_ASSISTANCE and artifact.status == "declined"
        for artifact in consent_artifacts
    )
    manual_profile = profile.manual_mode_profile
    if declined_recording:
        manual_profile = replace(
            manual_profile,
            transcription_mode=TranscriptionMode.DISABLED,
            transcript_persistence=TranscriptPersistence.SUPPRESSED,
        )
    if declined_ai and manual_profile.transcription_mode == TranscriptionMode.ACTIVE:
        manual_profile = replace(
            manual_profile,
            transcript_persistence=TranscriptPersistence.AUDIT_ONLY,
        )
    return manual_profile


def resolve_consent_capture(
    session: CallSession,
    profile: PolicyProfile,
    consent_type: ConsentType,
    granted: bool,
    prior_artifacts: list[ConsentArtifact],
) -> tuple[ConsentArtifact, CallState, CallMode]:
    if consent_type == ConsentType.RECORDING:
        artifact = ConsentArtifact(
            session_id=session.session_id,
            consent_type=ConsentType.RECORDING,
            status="granted" if granted else "declined",
            jurisdiction_policy_id=profile.policy_profile_id,
            captured_at=session.started_at,
            capture_mode="ai_led",
            storage_effect=_recording_storage_effect(granted),
        )
        if granted:
            next_state = (
                CallState.CONSENT_AI_ASSISTANCE
                if profile.require_ai_assistance_consent
                else CallState.IDENTITY_CAPTURE
            )
            return artifact, next_state, CallMode.AI_LED
        if profile.allow_ai_without_recording:
            next_state = (
                CallState.CONSENT_AI_ASSISTANCE
                if profile.require_ai_assistance_consent
                else CallState.IDENTITY_CAPTURE
            )
            return artifact, next_state, CallMode.AI_LED
        fallback_mode = CallMode.MANUAL if profile.allow_service_after_recording_decline else CallMode.HUMAN_TAKEOVER
        return artifact, CallState.CONSENT_DECLINED, fallback_mode

    prior_recording_declined = any(
        existing.consent_type == ConsentType.RECORDING and existing.status == "declined"
        for existing in prior_artifacts
    )
    artifact = ConsentArtifact(
        session_id=session.session_id,
        consent_type=ConsentType.AI_ASSISTANCE,
        status="granted" if granted else "declined",
        jurisdiction_policy_id=profile.policy_profile_id,
        captured_at=session.started_at,
        capture_mode="ai_led",
        storage_effect=_ai_storage_effect(granted),
    )
    if granted:
        return artifact, CallState.IDENTITY_CAPTURE, CallMode.AI_LED
    if prior_recording_declined and profile.allow_service_after_both_declined:
        return artifact, CallState.CONSENT_DECLINED, CallMode.MANUAL
    fallback_mode = CallMode.MANUAL if profile.allow_service_after_ai_decline else CallMode.HUMAN_TAKEOVER
    return artifact, CallState.CONSENT_DECLINED, fallback_mode


class PolicyEngine:
    def __init__(self, profile: PolicyProfile) -> None:
        self.profile = profile

    def decide(self, session: CallSession, template_id: str, requested_action: str) -> PolicyDecision:
        allowed = True
        reason_code = "allowed"
        if session.mode == CallMode.MANUAL:
            allowed = False
            reason_code = "manual_mode_tts_disabled"
        elif session.mode == CallMode.EMERGENCY and template_id != "emergency_redirect":
            allowed = False
            reason_code = "emergency_mode_template_locked"
        return PolicyDecision(
            decision_id=_new_id(),
            session_id=session.session_id,
            state=session.current_state,
            requested_action=requested_action,
            allowed=allowed,
            reason_code=reason_code,
            policy_profile_id=self.profile.policy_profile_id,
            template_id=template_id,
        )

