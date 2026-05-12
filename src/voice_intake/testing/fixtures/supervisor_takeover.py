from voice_intake.models import CallState, ConsentType, StirShakenAttestation, SupervisorInterventionType
from voice_intake.policy import PolicyProfile
from voice_intake.testing.call_runner import CallScript, ConsentGrant, ScriptStep, SupervisorAction

# Supervisor forces a takeover mid-intake → HUMAN_TAKEOVER state.
SUPERVISOR_TAKEOVER = CallScript(
    label="supervisor_forced_takeover",
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
            caller_text="My name is Maria",
            expected_state_after=CallState.DEMOGRAPHICS,
            expected_action_template="collect_field_prompt",
            llm_response={
                "template_id": "collect_field_prompt",
                "variables": {"field_label": "name"},
                "requested_transition": "demographics",
            },
        ),
        # Supervisor decides to take over (e.g., complex situation detected)
        ScriptStep(
            caller_text="",
            expected_state_after=CallState.HUMAN_TAKEOVER,
            supervisor_action=SupervisorAction(
                intervention_type=SupervisorInterventionType.FORCED_TAKEOVER,
                reason_code="complex_case",
                notes="Caller has a rare insurance situation requiring manual processing.",
            ),
        ),
    ],
)
