from voice_intake.models import CallState, ConsentType, StirShakenAttestation
from voice_intake.policy import PolicyProfile
from voice_intake.testing.call_runner import CallScript, ConsentGrant, ScriptStep

# Caller signals distress mid-intake; LLM correctly proposes emergency_redirect
# and transitions to EMERGENCY_EXIT.
EMERGENCY_EXIT = CallScript(
    label="emergency_exit_mid_intake",
    attestation=StirShakenAttestation.A,
    policy_profile=PolicyProfile(),
    steps=[
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
        ScriptStep(
            caller_text="My name is Alex",
            expected_state_after=CallState.DEMOGRAPHICS,
            expected_action_template="collect_field_prompt",
            llm_response={
                "template_id": "collect_field_prompt",
                "variables": {"field_label": "name"},
                "requested_transition": "demographics",
            },
        ),
        # LLM detects emergency in caller text; hold_message is the only template valid
        # in DEMOGRAPHICS that can carry a transition to EMERGENCY_EXIT.
        ScriptStep(
            caller_text="I think I am having a heart attack please help",
            expected_state_after=CallState.EMERGENCY_EXIT,
            expected_action_template="hold_message",
            llm_response={
                "template_id": "hold_message",
                "variables": {},
                "requested_transition": "emergency_exit",
            },
        ),
        # Now in EMERGENCY_EXIT: play the emergency redirect template and escalate.
        ScriptStep(
            caller_text="",
            expected_state_after=CallState.HUMAN_TAKEOVER,
            expected_action_template="emergency_redirect",
            llm_response={
                "template_id": "emergency_redirect",
                "variables": {},
                "requested_transition": "human_takeover",
            },
        ),
    ],
)
