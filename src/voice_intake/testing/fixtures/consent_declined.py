from voice_intake.models import CallState, ConsentType, StirShakenAttestation
from voice_intake.policy import PolicyProfile
from voice_intake.testing.call_runner import CallScript, ConsentGrant, ScriptStep

# Recording consent declined, AI assistance consent granted.
# With allow_ai_without_recording=True (default), the call continues in AI_LED mode.
CONSENT_RECORDING_DECLINED = CallScript(
    label="consent_recording_declined_ai_granted",
    attestation=StirShakenAttestation.A,
    policy_profile=PolicyProfile(
        require_recording_consent=True,
        require_ai_assistance_consent=True,
        allow_ai_without_recording=True,
    ),
    steps=[
        # Recording declined - still proceeds to AI consent because allow_ai_without_recording=True
        ScriptStep(
            caller_text="no",
            expected_state_after=CallState.CONSENT_AI_ASSISTANCE,
            consent_grant=ConsentGrant(ConsentType.RECORDING, False),
        ),
        # AI assistance granted → IDENTITY_CAPTURE
        ScriptStep(
            caller_text="yes",
            expected_state_after=CallState.IDENTITY_CAPTURE,
            consent_grant=ConsentGrant(ConsentType.AI_ASSISTANCE, True),
        ),
        # Call proceeds normally from here
        ScriptStep(
            caller_text="My name is John Smith",
            expected_state_after=CallState.DEMOGRAPHICS,
            expected_action_template="collect_field_prompt",
            llm_response={
                "template_id": "collect_field_prompt",
                "variables": {"field_label": "name"},
                "requested_transition": "demographics",
            },
        ),
    ],
)

# Both consents declined → CONSENT_DECLINED state → MANUAL_MODE
BOTH_CONSENTS_DECLINED = CallScript(
    label="both_consents_declined",
    attestation=StirShakenAttestation.A,
    policy_profile=PolicyProfile(
        require_recording_consent=True,
        require_ai_assistance_consent=True,
        allow_ai_without_recording=True,
        allow_service_after_both_declined=True,
    ),
    steps=[
        ScriptStep(
            caller_text="no",
            expected_state_after=CallState.CONSENT_AI_ASSISTANCE,
            consent_grant=ConsentGrant(ConsentType.RECORDING, False),
        ),
        ScriptStep(
            caller_text="no",
            expected_state_after=CallState.CONSENT_DECLINED,
            consent_grant=ConsentGrant(ConsentType.AI_ASSISTANCE, False),
        ),
    ],
)
