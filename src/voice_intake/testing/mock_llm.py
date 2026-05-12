from __future__ import annotations

from uuid import uuid4

from voice_intake.models import CallState, ModelProposal


def _new_id() -> str:
    return str(uuid4())


_FALLBACK_BY_STATE: dict[str, tuple[str, dict[str, str], str]] = {
    "opening": ("opening_disclosure", {}, "consent_recording"),
    "consent_recording": ("recording_consent_prompt", {}, "consent_ai_assistance"),
    "consent_ai_assistance": ("ai_assistance_consent_prompt", {}, "identity_capture"),
    "consent_declined": ("hold_message", {}, "identity_capture"),
    "identity_capture": (
        "collect_field_prompt",
        {"field_label": "date of birth"},
        "demographics",
    ),
    "demographics": (
        "collect_field_prompt",
        {"field_label": "address"},
        "insurance",
    ),
    "insurance": (
        "collect_field_prompt",
        {"field_label": "insurance"},
        "reason_for_visit",
    ),
    "reason_for_visit": (
        "collect_field_prompt",
        {"field_label": "reason for visit"},
        "readback",
    ),
    "readback": (
        "readback_confirmation",
        {"field_label": "date of birth", "masked_value": "****1985"},
        "disposition",
    ),
    "disposition": ("closing_prompt", {}, "close"),
    "manual_mode": ("hold_message", {}, "human_takeover"),
    "human_takeover": ("hold_message", {}, "close"),
    "emergency_exit": ("emergency_redirect", {}, "close"),
    # Note: "close" intentionally omitted - CLOSE has no allowed transitions, and
    # the previous "intake_complete" template doesn't exist in DEFAULT_TEMPLATE_BUNDLE.
}


class MockLLM:
    def __init__(self, script: list[dict] | None = None) -> None:
        self._queue: list[dict] = list(script) if script else []

    def propose(self, prompt: dict[str, object], source_turn_id: str) -> ModelProposal:
        if self._queue:
            entry = self._queue.pop(0)
        else:
            current = str(prompt.get("current_state", "opening"))
            template_id, variables, next_state = _FALLBACK_BY_STATE.get(
                current,
                ("hold_message", {}, current),
            )
            entry = {
                "template_id": template_id,
                "variables": variables,
                "requested_transition": next_state,
                "model_version": "mock-v1",
            }
        return ModelProposal(
            proposal_id=_new_id(),
            source_turn_id=source_turn_id,
            template_id=entry["template_id"],
            variables=entry.get("variables", {}),
            requested_transition=CallState(entry["requested_transition"]),
            model_version=entry.get("model_version", "mock-v1"),
        )
