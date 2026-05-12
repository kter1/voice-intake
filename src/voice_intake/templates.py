from __future__ import annotations

import re
from dataclasses import dataclass, field
from uuid import uuid4

from .models import CallState


def _new_id() -> str:
    return str(uuid4())


@dataclass(slots=True)
class TemplateVariableSpec:
    name: str
    pattern: str
    masked_only: bool = False

    def validate(self, value: str) -> bool:
        if self.masked_only and not any(mask in value for mask in ("*", "X", "x")):
            return False
        return re.fullmatch(self.pattern, value) is not None


@dataclass(slots=True)
class Template:
    template_id: str
    allowed_states: frozenset[CallState]
    allowed_variables: dict[str, TemplateVariableSpec]
    interruptible: bool
    content: str
    # When False, the template is reserved for system/timeout fallback paths
    # (e.g. hold_message) and is not exposed to the LLM as a selectable action.
    # The validator continues to accept it for system-emitted proposals.
    model_facing: bool = True


@dataclass(slots=True)
class TemplateBundle:
    version: str
    templates: dict[str, Template]
    approved_by: tuple[str, ...] = field(default_factory=tuple)
    approved_event_id: str | None = None

    @property
    def approved(self) -> bool:
        return bool(self.approved_by)

    def publish(self, reviewer_id: str) -> "TemplateBundle":
        if not reviewer_id:
            raise ValueError("reviewer_id is required")
        approved_by = tuple(sorted(set((*self.approved_by, reviewer_id))))
        return TemplateBundle(
            version=self.version,
            templates=self.templates,
            approved_by=approved_by,
            approved_event_id=_new_id(),
        )

    def template_for(self, template_id: str) -> Template | None:
        return self.templates.get(template_id)


DEFAULT_TEMPLATE_BUNDLE = TemplateBundle(
    version="v1",
    templates={
        "opening_disclosure": Template(
            template_id="opening_disclosure",
            allowed_states=frozenset({CallState.OPENING}),
            allowed_variables={},
            interruptible=False,
            content="You're speaking with an AI assistant. How can I help?",
        ),
        "recording_consent_prompt": Template(
            template_id="recording_consent_prompt",
            allowed_states=frozenset({CallState.CONSENT_RECORDING}),
            allowed_variables={},
            interruptible=False,
            content="Do you consent to call recording where required by law?",
        ),
        "ai_assistance_consent_prompt": Template(
            template_id="ai_assistance_consent_prompt",
            allowed_states=frozenset({CallState.CONSENT_AI_ASSISTANCE}),
            allowed_variables={},
            interruptible=False,
            content="Do you consent to speaking with an AI assistant supervised by staff?",
        ),
        "collect_field_prompt": Template(
            template_id="collect_field_prompt",
            allowed_states=frozenset(
                {
                    CallState.IDENTITY_CAPTURE,
                    CallState.DEMOGRAPHICS,
                    CallState.INSURANCE,
                    CallState.REASON_FOR_VISIT,
                }
            ),
            allowed_variables={
                "field_label": TemplateVariableSpec("field_label", r"[A-Za-z _-]{2,40}"),
            },
            interruptible=True,
            content="Please tell me your {{field_label}}.",
        ),
        "readback_confirmation": Template(
            template_id="readback_confirmation",
            allowed_states=frozenset({CallState.READBACK}),
            allowed_variables={
                "field_label": TemplateVariableSpec("field_label", r"[A-Za-z _-]{2,40}"),
                "masked_value": TemplateVariableSpec(
                    "masked_value", r"[\*Xx0-9 -]{4,24}", masked_only=True
                ),
            },
            interruptible=True,
            content="I have {{field_label}} ending in {{masked_value}}. Is that correct?",
        ),
        "identity_threshold_denial": Template(
            template_id="identity_threshold_denial",
            allowed_states=frozenset({CallState.IDENTITY_CAPTURE}),
            allowed_variables={},
            interruptible=False,
            content="I cannot continue until a staff member joins the call.",
        ),
        "hold_message": Template(
            template_id="hold_message",
            allowed_states=frozenset(
                state for state in CallState if state != CallState.CLOSE
            ),
            allowed_variables={},
            interruptible=True,
            content="Please hold one moment.",
            model_facing=False,
        ),
        "emergency_redirect": Template(
            template_id="emergency_redirect",
            allowed_states=frozenset(
                state for state in CallState if state != CallState.CLOSE
            ),
            allowed_variables={},
            interruptible=False,
            content=(
                "This sounds like an emergency. Please dial 911 now or seek emergency care "
                "immediately. I'll stop the intake and alert staff."
            ),
        ),
        "closing_prompt": Template(
            template_id="closing_prompt",
            allowed_states=frozenset(
                {
                    CallState.CONSENT_DECLINED,
                    CallState.DISPOSITION,
                    CallState.MANUAL_MODE,
                    CallState.HUMAN_TAKEOVER,
                    CallState.EMERGENCY_EXIT,
                }
            ),
            allowed_variables={},
            interruptible=True,
            content="Thank you. Your intake has been recorded for staff follow-up.",
        ),
        # ── Demo scheduling-flow templates ────────────────────────────────────
        # allowed_states is the SOURCE state - where the assistant IS when it
        # emits the template (checked by ProposalValidator before the transition).
        "appointment_reason_prompt": Template(
            template_id="appointment_reason_prompt",
            allowed_states=frozenset({CallState.OPENING, CallState.REASON_FOR_VISIT}),
            allowed_variables={},
            interruptible=True,
            content="What is the reason for the visit?",
        ),
        "patient_name_prompt": Template(
            template_id="patient_name_prompt",
            allowed_states=frozenset({CallState.REASON_FOR_VISIT, CallState.IDENTITY_CAPTURE}),
            allowed_variables={},
            interruptible=True,
            content="What is your full name?",
        ),
        "dob_prompt": Template(
            template_id="dob_prompt",
            allowed_states=frozenset({CallState.IDENTITY_CAPTURE, CallState.DEMOGRAPHICS}),
            allowed_variables={},
            interruptible=True,
            content="What is your date of birth?",
        ),
        "dob_clarification": Template(
            template_id="dob_clarification",
            allowed_states=frozenset({CallState.DEMOGRAPHICS}),
            allowed_variables={},
            interruptible=True,
            content="I didn't catch the date of birth. Please say it like month, day, year.",
        ),
        "insurance_prompt": Template(
            template_id="insurance_prompt",
            allowed_states=frozenset({CallState.DEMOGRAPHICS, CallState.INSURANCE}),
            allowed_variables={},
            interruptible=True,
            content="What insurance should we use for this appointment?",
        ),
        "insurance_in_network": Template(
            template_id="insurance_in_network",
            allowed_states=frozenset({CallState.INSURANCE}),
            allowed_variables={
                "insurance_name": TemplateVariableSpec(
                    "insurance_name", r"[A-Za-z0-9 \-&'.]{1,80}"
                ),
            },
            interruptible=True,
            content=(
                "I'm showing {{insurance_name}} as in network. "
                "I can schedule the appointment. Is that okay?"
            ),
        ),
        "insurance_out_of_network": Template(
            template_id="insurance_out_of_network",
            allowed_states=frozenset({CallState.INSURANCE}),
            allowed_variables={
                "insurance_name": TemplateVariableSpec(
                    "insurance_name", r"[A-Za-z0-9 \-&'.]{1,80}"
                ),
            },
            interruptible=True,
            content=(
                "I'm showing {{insurance_name}} as out of network for this demo. "
                "You may have higher out-of-pocket costs. Do you still want to continue?"
            ),
        ),
        "insurance_unknown": Template(
            template_id="insurance_unknown",
            allowed_states=frozenset({CallState.INSURANCE}),
            allowed_variables={
                "insurance_name": TemplateVariableSpec(
                    "insurance_name", r"[A-Za-z0-9 \-&'.]{1,80}"
                ),
            },
            interruptible=True,
            content=(
                "I can't verify {{insurance_name}} in this demo. "
                "I can still schedule the request for staff review. Is that okay?"
            ),
        ),
        "insurance_not_provided": Template(
            template_id="insurance_not_provided",
            allowed_states=frozenset({CallState.INSURANCE}),
            allowed_variables={},
            interruptible=True,
            content=(
                "I can continue without insurance, but staff may need to review "
                "the request. Is that okay?"
            ),
        ),
        "appointment_scheduled": Template(
            template_id="appointment_scheduled",
            allowed_states=frozenset({CallState.READBACK}),
            allowed_variables={},
            interruptible=False,
            content=(
                "Your appointment has been scheduled. "
                "A staff member can review the intake summary if needed."
            ),
        ),
        "appointment_request_created": Template(
            template_id="appointment_request_created",
            allowed_states=frozenset({CallState.READBACK}),
            allowed_variables={},
            interruptible=False,
            content=(
                "Your appointment request has been created for staff review. "
                "A staff member can review the intake summary if needed."
            ),
        ),
        "human_handoff": Template(
            template_id="human_handoff",
            allowed_states=frozenset(
                {
                    CallState.OPENING,
                    CallState.REASON_FOR_VISIT,
                    CallState.IDENTITY_CAPTURE,
                    CallState.DEMOGRAPHICS,
                    CallState.INSURANCE,
                    CallState.READBACK,
                    CallState.HUMAN_TAKEOVER,
                }
            ),
            allowed_variables={},
            interruptible=False,
            content="Of course. I'll alert staff to continue from here.",
        ),
        "cannot_schedule_without_identity": Template(
            template_id="cannot_schedule_without_identity",
            allowed_states=frozenset(
                {
                    CallState.IDENTITY_CAPTURE,
                    CallState.DEMOGRAPHICS,
                    CallState.INSURANCE,
                }
            ),
            allowed_variables={},
            interruptible=False,
            content=(
                "I can't schedule without that information, "
                "but I can alert staff to continue."
            ),
        ),
        "field_explanation_for_reason": Template(
            template_id="field_explanation_for_reason",
            allowed_states=frozenset({CallState.REASON_FOR_VISIT}),
            allowed_variables={},
            interruptible=True,
            content=(
                "Staff use this to understand what type of appointment you need. "
                "What is the reason for the visit?"
            ),
        ),
        "field_explanation_for_name": Template(
            template_id="field_explanation_for_name",
            allowed_states=frozenset({CallState.IDENTITY_CAPTURE}),
            allowed_variables={},
            interruptible=True,
            content=(
                "Staff use this to match the correct patient record and prepare "
                "the appointment request. What is your full name?"
            ),
        ),
        "field_explanation_for_dob": Template(
            template_id="field_explanation_for_dob",
            allowed_states=frozenset({CallState.DEMOGRAPHICS}),
            allowed_variables={},
            interruptible=True,
            content=(
                "Staff use this to match the correct patient record and avoid mixing up "
                "patients with similar names. What is your date of birth?"
            ),
        ),
        "field_explanation_for_insurance": Template(
            template_id="field_explanation_for_insurance",
            allowed_states=frozenset({CallState.INSURANCE}),
            allowed_variables={},
            interruptible=True,
            content=(
                "Staff use this to check the demo network status and prepare the "
                "appointment request. What insurance should we use for this appointment?"
            ),
        ),
        "demo_capabilities_prompt": Template(
            template_id="demo_capabilities_prompt",
            allowed_states=frozenset({CallState.OPENING}),
            allowed_variables={},
            interruptible=True,
            content=(
                "I can help with a demo appointment intake, collect scheduling details, "
                "answer configured practice-policy questions when available, and route emergencies "
                "or staff handoff when needed. To start scheduling, tell me the reason for the visit."
            ),
        ),
    },
).publish("compliance-reviewer-1")

