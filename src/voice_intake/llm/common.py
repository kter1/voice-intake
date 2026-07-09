from __future__ import annotations

import copy
import json
import re
from typing import Literal

from voice_intake.models import CallState
from voice_intake.normalization import DIGIT_BLOCK_RE, DOB_RE, PHONE_RE

PromptProfile = Literal["default", "demo", "recruiter_demo"]

_SYSTEM_PROMPT = """\
You are a supervised AI voice intake assistant for a healthcare practice.
Your ONLY output is a call to the `propose_action` tool - never plain text.
Rules:
- Follow the state machine: only request transitions that are valid from the current state.
- When the prompt includes an `allowed_transitions` list, it is AUTHORITATIVE.
  Copy `requested_transition` EXACTLY from that list. Never request a transition
  that is not in `allowed_transitions`.
- In recruiter demo mode, do not request consent_recording, consent_ai_assistance,
  or consent_declined unless they appear in `allowed_transitions`.
- Do not put raw PHI in variables. Use masked/placeholder values.
- If the caller reports an emergency (chest pain, difficulty breathing, suicidal ideation),
  propose a transition to emergency_exit immediately.
- Use the [CONTEXT] blocks below to answer questions about practice policies accurately.
- If a [PATIENT_CONTEXT] block is present, use it to pre-populate known fields.
- If an [ELIGIBILITY] block is present, include coverage info when relevant.
"""

_RECRUITER_DEMO_PROMPT = """\
Recruiter demo profile:
- Stay within appointment scheduling intake only.
- Use only the available_templates list and provided context.
- Do not invent fields, appointment slots, eligibility details, coverage guarantees, diagnoses, or treatment advice.
- If the caller asks outside scheduling scope, choose the safest in-scope template or escalate when required.
- Emergency or safety language must still trigger the emergency/safety path.
- This is a synthetic demo interaction; do not describe it as clinical proof, compliance evidence, or deployment evidence.
- You may include `spoken_response`: one or two short conversational sentences
  carrying the SAME intent as your chosen template. Acknowledge what the caller
  actually said, then ask or state what the template asks or states. Plain prose
  only - no digits runs, no markup, no PHI, no diagnoses or treatment advice.
  When unsure, omit it and the approved template text is spoken instead.
"""

_RAG_KEYS = {
    "knowledge_context",
    "patient_context",
    "eligibility_context",
    "prior_calls_context",
}

_INJECTION_PATTERNS = [
    r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\b",
    r"(?i)\bsystem\s*:?\s*you\s+are\b",
    r"<\s*/?\s*(system|user|assistant)\s*>",
    r"\[INST\]|\[/INST\]",
    r"###\s*(Instruction|System|Human|Assistant)",
]

# PHI spans are redacted here - at the boundary where text leaves the process
# for a model provider - not in build_model_prompt, whose output is also
# consumed by in-process deterministic components (DemoRouter field
# extraction) that need the raw spans. Order matters: phone numbers first so
# separator-free ones aren't half-consumed as generic digit blocks.
_PHI_SPAN_PATTERNS = (
    ("CALLBACK_NUMBER", PHONE_RE),
    ("DOB", DOB_RE),
    ("MEMBER_ID", DIGIT_BLOCK_RE),
)

OAI_PROPOSE_TOOL = {
    "type": "function",
    "function": {
        "name": "propose_action",
        "description": (
            "Select the next voice action for the caller interaction. "
            "Use ONLY this tool - never respond with raw text. "
            "Never put raw PHI (full phone numbers, SSNs, DOBs) in variables; use masked values."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "template_id": {
                    "type": "string",
                    "description": "The template ID from the available_templates list.",
                },
                "variables": {
                    "type": "object",
                    "description": "Template variables. Use masked values only (no raw PHI).",
                },
                "requested_transition": {
                    "type": "string",
                    "description": (
                        "The target CallState after this action. MUST be copied "
                        "exactly from the prompt's `allowed_transitions` list when "
                        "that list is present. Never request a transition not in "
                        "that list."
                    ),
                },
                "spoken_response": {
                    "type": "string",
                    "description": (
                        "Optional: one or two short conversational sentences with "
                        "the SAME intent as the chosen template. Plain prose only - "
                        "no markup, no digit runs, no PHI, no diagnoses or treatment "
                        "advice. Omit when unsure; the template text is spoken then."
                    ),
                },
            },
            "required": ["template_id", "variables", "requested_transition"],
        },
    },
}


def _sanitize_for_llm(text: str) -> str:
    for pattern in _INJECTION_PATTERNS:
        text = re.sub(pattern, "[REDACTED]", text)
    for label, phi_pattern in _PHI_SPAN_PATTERNS:
        text = phi_pattern.sub(f"[{label}]", text)
    return text


def sanitize_prompt(prompt: dict) -> dict:
    clean = copy.deepcopy(prompt)
    recent_turns = clean.get("recent_turns") or []
    if isinstance(recent_turns, list):
        for turn in recent_turns:
            if isinstance(turn, dict) and isinstance(turn.get("transcript"), str):
                turn["transcript"] = _sanitize_for_llm(turn["transcript"])
    if isinstance(clean.get("transcript"), str):
        clean["transcript"] = _sanitize_for_llm(clean["transcript"])
    return clean


def build_system_prompt(
    prompt: dict,
    prompt_profile: PromptProfile | str = "default",
) -> str:
    parts = [_SYSTEM_PROMPT]
    if prompt_profile in {"demo", "recruiter_demo"}:
        parts.append(_RECRUITER_DEMO_PROMPT)
    if prompt.get("knowledge_context"):
        chunks = prompt["knowledge_context"]
        lines = "\n".join(f"- {c['text'][:300]}" for c in chunks)
        parts.append(f"\n[CONTEXT]\n{lines}")
    if prompt.get("patient_context"):
        parts.append(f"\n[PATIENT_CONTEXT]\n{json.dumps(prompt['patient_context'])}")
    if prompt.get("eligibility_context"):
        parts.append(f"\n[ELIGIBILITY]\n{json.dumps(prompt['eligibility_context'])}")
    if prompt.get("prior_calls_context"):
        parts.append(f"\n[PRIOR_CONTEXT]\n{json.dumps(prompt['prior_calls_context'])}")
    return "\n".join(parts)


def build_user_message(prompt: dict) -> str:
    sanitized = sanitize_prompt(prompt)
    sanitized = {k: v for k, v in sanitized.items() if k not in _RAG_KEYS}
    return json.dumps(sanitized, default=str)


def coerce_transition(raw: CallState | str | None) -> CallState:
    if raw is None or raw == "":
        raise ValueError("requested_transition is missing or empty")
    if isinstance(raw, CallState):
        return raw
    return CallState(str(raw))
