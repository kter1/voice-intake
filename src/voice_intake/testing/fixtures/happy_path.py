from voice_intake.models import CallDisposition, CallState, ConsentType, StirShakenAttestation
from voice_intake.policy import PolicyProfile
from voice_intake.testing.call_runner import CallScript, ConsentGrant, ScriptStep

HAPPY_PATH = CallScript(
    label="happy_path",
    attestation=StirShakenAttestation.A,
    policy_profile=PolicyProfile(),
    expected_final_disposition=None,
    steps=[
        # Both consents granted
        ScriptStep(
            caller_text="yes",
            expected_state_after=CallState.CONSENT_AI_ASSISTANCE,
            consent_grant=ConsentGrant(ConsentType.RECORDING, True),
        ),
        ScriptStep(
            caller_text="yes",
            expected_state_after=CallState.IDENTITY_CAPTURE,
            consent_grant=ConsentGrant(ConsentType.AI_ASSISTANCE, True),
        ),
        # Collect name → DEMOGRAPHICS
        ScriptStep(
            caller_text="My name is Jane Doe",
            expected_state_after=CallState.DEMOGRAPHICS,
            expected_action_template="collect_field_prompt",
            llm_response={
                "template_id": "collect_field_prompt",
                "variables": {"field_label": "name"},
                "requested_transition": "demographics",
            },
        ),
        # Collect DOB → INSURANCE
        ScriptStep(
            caller_text="My date of birth is January 1st 1980",
            expected_state_after=CallState.INSURANCE,
            expected_action_template="collect_field_prompt",
            llm_response={
                "template_id": "collect_field_prompt",
                "variables": {"field_label": "date of birth"},
                "requested_transition": "insurance",
            },
        ),
        # Collect insurance → REASON_FOR_VISIT
        ScriptStep(
            caller_text="I have BlueCross",
            expected_state_after=CallState.REASON_FOR_VISIT,
            expected_action_template="collect_field_prompt",
            llm_response={
                "template_id": "collect_field_prompt",
                "variables": {"field_label": "insurance plan"},
                "requested_transition": "reason_for_visit",
            },
        ),
        # Collect reason → READBACK
        ScriptStep(
            caller_text="I have a persistent cough",
            expected_state_after=CallState.READBACK,
            expected_action_template="collect_field_prompt",
            llm_response={
                "template_id": "collect_field_prompt",
                "variables": {"field_label": "reason for visit"},
                "requested_transition": "readback",
            },
        ),
        # Readback confirmation → DISPOSITION
        ScriptStep(
            caller_text="yes that is correct",
            expected_state_after=CallState.DISPOSITION,
            expected_action_template="readback_confirmation",
            llm_response={
                "template_id": "readback_confirmation",
                "variables": {
                    "field_label": "member ID",
                    "masked_value": "*****6789",
                },
                "requested_transition": "disposition",
            },
        ),
        # Close the call
        ScriptStep(
            caller_text="thank you",
            expected_state_after=CallState.CLOSE,
            expected_action_template="hold_message",
            llm_response={
                "template_id": "hold_message",
                "variables": {},
                "requested_transition": "close",
            },
        ),
    ],
)
