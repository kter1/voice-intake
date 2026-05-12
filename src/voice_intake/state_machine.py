from __future__ import annotations

from .models import CallState


ALLOWED_TRANSITIONS: dict[CallState, frozenset[CallState]] = {
    CallState.OPENING: frozenset(
        {
            CallState.OPENING,  # self-loop: meta/help question, stay at opening
            CallState.CONSENT_RECORDING,
            CallState.CONSENT_AI_ASSISTANCE,
            CallState.REASON_FOR_VISIT,  # demo flow: skip consent, go straight to scheduling
            CallState.HUMAN_TAKEOVER,
            CallState.EMERGENCY_EXIT,
        }
    ),
    CallState.CONSENT_RECORDING: frozenset(
        {
            CallState.CONSENT_AI_ASSISTANCE,
            CallState.CONSENT_DECLINED,
            CallState.HUMAN_TAKEOVER,
            CallState.EMERGENCY_EXIT,
        }
    ),
    CallState.CONSENT_AI_ASSISTANCE: frozenset(
        {
            CallState.IDENTITY_CAPTURE,
            CallState.CONSENT_DECLINED,
            CallState.HUMAN_TAKEOVER,
            CallState.EMERGENCY_EXIT,
        }
    ),
    CallState.CONSENT_DECLINED: frozenset(
        {
            CallState.IDENTITY_CAPTURE,
            CallState.MANUAL_MODE,
            CallState.HUMAN_TAKEOVER,
            CallState.CLOSE,
            CallState.EMERGENCY_EXIT,
        }
    ),
    CallState.IDENTITY_CAPTURE: frozenset(
        {
            CallState.IDENTITY_CAPTURE,  # self-loop: "why?" reply
            CallState.DEMOGRAPHICS,
            CallState.HUMAN_TAKEOVER,
            CallState.EMERGENCY_EXIT,
            CallState.MANUAL_MODE,
        }
    ),
    CallState.DEMOGRAPHICS: frozenset(
        {
            CallState.DEMOGRAPHICS,  # self-loop: DOB parse failure or "why?" reply
            CallState.INSURANCE,
            CallState.HUMAN_TAKEOVER,
            CallState.EMERGENCY_EXIT,
            CallState.MANUAL_MODE,
        }
    ),
    CallState.INSURANCE: frozenset(
        {
            CallState.INSURANCE,  # self-loop: "why?" reply
            CallState.REASON_FOR_VISIT,
            CallState.READBACK,  # demo flow: insurance → readback (confirmation)
            CallState.HUMAN_TAKEOVER,
            CallState.EMERGENCY_EXIT,
            CallState.MANUAL_MODE,
        }
    ),
    CallState.REASON_FOR_VISIT: frozenset(
        {
            CallState.REASON_FOR_VISIT,  # self-loop: "why?" reply or generic re-ask
            CallState.IDENTITY_CAPTURE,  # demo flow: reason → name
            CallState.READBACK,
            CallState.HUMAN_TAKEOVER,
            CallState.EMERGENCY_EXIT,
            CallState.MANUAL_MODE,
        }
    ),
    CallState.READBACK: frozenset(
        {
            CallState.READBACK,  # self-loop: "why?" or repeated confirmation ask
            CallState.DISPOSITION,
            CallState.HUMAN_TAKEOVER,
            CallState.EMERGENCY_EXIT,
            CallState.MANUAL_MODE,
        }
    ),
    CallState.DISPOSITION: frozenset(
        {
            CallState.CLOSE,
            CallState.HUMAN_TAKEOVER,
            CallState.EMERGENCY_EXIT,
        }
    ),
    CallState.MANUAL_MODE: frozenset(
        {
            CallState.IDENTITY_CAPTURE,
            CallState.DEMOGRAPHICS,
            CallState.INSURANCE,
            CallState.REASON_FOR_VISIT,
            CallState.READBACK,
            CallState.DISPOSITION,
            CallState.CLOSE,
            CallState.HUMAN_TAKEOVER,
            CallState.EMERGENCY_EXIT,
        }
    ),
    CallState.HUMAN_TAKEOVER: frozenset(
        {
            CallState.DISPOSITION,
            CallState.CLOSE,
            CallState.EMERGENCY_EXIT,
        }
    ),
    CallState.EMERGENCY_EXIT: frozenset(
        {
            CallState.HUMAN_TAKEOVER,
            CallState.DISPOSITION,
            CallState.CLOSE,
        }
    ),
    CallState.CLOSE: frozenset(),
}


def allowed_targets(state: CallState) -> frozenset[CallState]:
    return ALLOWED_TRANSITIONS[state]


def can_transition(current_state: CallState, requested_state: CallState) -> bool:
    return requested_state in ALLOWED_TRANSITIONS[current_state]


def assert_transition(current_state: CallState, requested_state: CallState) -> None:
    if not can_transition(current_state, requested_state):
        raise ValueError(f"illegal transition: {current_state} -> {requested_state}")

