from uuid import uuid4

from voice_intake.models import CallState, ConsentType, StirShakenAttestation
from voice_intake.policy import PolicyProfile
from voice_intake.testing.call_runner import CallScript, ConsentGrant, ScriptStep

# Shared turn_id simulates two proposals from the same caller utterance.
# The validator rejects both (DISALLOWED_TRANSITION) → HUMAN_TAKEOVER.
_SHARED_TURN = str(uuid4())

DOUBLE_REJECT = CallScript(
    label="double_reject_forces_human_takeover",
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
        # First bad proposal from this turn: IDENTITY_CAPTURE → CLOSE (disallowed)
        ScriptStep(
            caller_text="My name is Bob",
            expected_state_after=CallState.IDENTITY_CAPTURE,
            expected_action_template=None,
            expect_rejection=True,
            turn_id=_SHARED_TURN,
            llm_response={
                "template_id": "closing_prompt",  # not valid in IDENTITY_CAPTURE
                "variables": {},
                "requested_transition": "close",
            },
        ),
        # Second bad proposal from the SAME turn → double-reject → HUMAN_TAKEOVER
        ScriptStep(
            caller_text="My name is Bob",
            expected_state_after=CallState.HUMAN_TAKEOVER,
            expected_action_template=None,
            expect_rejection=True,
            turn_id=_SHARED_TURN,
            llm_response={
                "template_id": "closing_prompt",
                "variables": {},
                "requested_transition": "close",
            },
        ),
    ],
)
