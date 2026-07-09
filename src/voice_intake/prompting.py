from __future__ import annotations

import re

from .boundary import (
    PromptBoundaryPolicy,
    inspect_prompt_payload,
    sanitize_prompt_payload,
)
from .models import CallSession, CallState, FieldCandidate, VoiceTurn
from .templates import Template


def mask_value(field_name: str, raw_value: str) -> str:
    if field_name in {"callback_number", "member_id"}:
        digits = re.sub(r"\D", "", raw_value)
        if len(digits) <= 4:
            return "*" * len(digits)
        return "*" * (len(digits) - 4) + digits[-4:]
    if field_name == "dob":
        digits = re.sub(r"\D", "", raw_value)
        return digits[-4:] if len(digits) >= 4 else "****"
    if not raw_value:
        return "masked"
    return raw_value[0] + "*" * max(len(raw_value) - 1, 1)


def _redact_text(text: str, field_candidates: list[FieldCandidate]) -> str:
    redacted = text
    for candidate in field_candidates:
        if candidate.candidate_value:
            redacted = re.sub(
                re.escape(candidate.candidate_value),
                f"[{candidate.field_name.upper()}]",
                redacted,
                flags=re.IGNORECASE,
            )
    return redacted


def build_model_prompt(
    session: CallSession,
    recent_turns: list[VoiceTurn],
    field_candidates: list[FieldCandidate],
    available_templates: list[Template],
    confirmation_field: FieldCandidate | None = None,
    knowledge_chunks: list[dict] | None = None,
    patient_context: dict | None = None,
    eligibility_context: dict | None = None,
    prior_calls_context: list[dict] | None = None,
    boundary_policy: PromptBoundaryPolicy | None = None,
    allowed_transitions: frozenset[CallState] | list[CallState] | None = None,
) -> dict[str, object]:
    slot_status = [
        {
            "field_name": candidate.field_name,
            "verification_status": candidate.verification_status.value,
        }
        for candidate in field_candidates
    ]
    # Candidate-value redaction only: in-process deterministic consumers
    # (DemoRouter field extraction, safety prerouter) need the raw spans.
    # PHI span redaction for text leaving the process happens at the LLM
    # boundary in llm/common.sanitize_prompt, applied to every turn.
    recent_context = [
        {
            "speaker": turn.speaker.value,
            "transcript": _redact_text(turn.transcript, field_candidates),
        }
        for turn in recent_turns[-4:]
    ]
    prompt: dict[str, object] = {
        "session_id": session.session_id,
        "current_state": session.current_state.value,
        "slot_status": slot_status,
        "recent_turns": recent_context,
        "available_templates": [template.template_id for template in available_templates],
    }
    if allowed_transitions is not None:
        prompt["allowed_transitions"] = [
            state.value for state in allowed_transitions
        ]
    if confirmation_field is not None:
        prompt["confirmation_context"] = {
            "field_name": confirmation_field.field_name,
            "masked_value": mask_value(
                confirmation_field.field_name, confirmation_field.candidate_value
            ),
        }
    if knowledge_chunks:
        prompt["knowledge_context"] = knowledge_chunks
    if patient_context:
        prompt["patient_context"] = patient_context
    if eligibility_context:
        prompt["eligibility_context"] = eligibility_context
    if prior_calls_context:
        prompt["prior_calls_context"] = prior_calls_context
    if boundary_policy is not None:
        prompt = sanitize_prompt_payload(prompt, field_candidates, boundary_policy)
        inspect_prompt_payload(prompt, field_candidates, boundary_policy)
    return prompt
