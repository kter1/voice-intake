"""DemoRouter - deterministic appointment-scheduling intent router for the public demo.

Acts as a drop-in replacement for MockLLM / a remote LLM when DEMO_ROUTER=true.
It is an app-state singleton (set in lifespan, injected via get_demo_router) so
pending appointment patches survive across propose() and commit()/discard().

Architecture contract
─────────────────────
propose() is called BEFORE ProposalValidator runs.  Mutating AppointmentRequest
storage inside propose() would violate the repo's "no business-state mutation
before validation" invariant.

Pattern used here:
  1. propose() returns a ModelProposal AND records a pending AppointmentPatch
     keyed by proposal_id in an in-memory dict.
  2. /turn calls commit(proposal_id) on validator acceptance - applies the patch.
  3. /turn calls discard(proposal_id) on rejection - drops the patch silently.
  4. Patches older than PATCH_TTL_SECONDS are pruned on every propose() call.

proposal_id consistency invariant
──────────────────────────────────
The same proposal_id must flow:
  turns.py → propose_next_action(proposal_id=X)
           → DemoRouter.propose(..., proposal_id=X)
              → pending_patches[X] = patch
              → ModelProposal.proposal_id == X
  /turn → demo_router.commit(proposal.proposal_id)   # X again
         → store.upsert_appointment_request(patch)

Any drift between these IDs causes silent no-ops on the appointment table.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy.orm import sessionmaker

from .db.store import SQLAuditStore
from .intent_patterns import RE_EMERGENCY as _RE_EMERGENCY
from .intent_patterns import RE_HUMAN as _RE_HUMAN
from .intent_patterns import compile_phrases
from .models import (
    AppointmentRequest,
    AppointmentStatus,
    CallState,
    InsuranceNetworkStatus,
    ModelProposal,
)

PATCH_TTL_SECONDS: int = 600  # prune pending patches older than 10 minutes
PATCH_MAX_SIZE: int = 1000     # hard-cap; clear all if exceeded (runaway safeguard)

_DEMO_MODEL_VERSION = "demo-router-v1"


def _new_id() -> str:
    return str(uuid4())


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ── Intent keyword sets ───────────────────────────────────────────────────────

_WHY_PHRASES = [
    r"\bwhy\b",
    r"why\s+do\s+you\s+need",
    r"what\s+for\b",
    r"why\s+are\s+you\s+asking",
]

_AFFIRMATION_PHRASES = [
    r"\byes\b",
    r"\byeah\b",
    r"\bcorrect\b",
    r"that'?s?\s+fine",
    r"\bokay\b",
    r"\bok\b",
    r"\bsure\b",
    r"sounds\s+good",
    r"go\s+ahead",
]

_DENIAL_PHRASES = [
    r"\bno\b",
    r"\bnope\b",
    r"\bincorrect\b",
    r"that'?s?\s+wrong",
    # Negation phrases - checked before affirmation in classify_intent.
    # Targeted on purpose: bare \bdon't\b / do\s+not are NOT in this list,
    # because they would steal refusal phrases like "I don't want to give that"
    # which need to stay REFUSAL for identity-state routing to work.
    r"not\s+okay",
    r"not\s+correct",
    r"not\s+(right|fine|sure)",
    r"do\s+not\s+(schedule|continue|book|make)",
    r"don'?t\s+(schedule|continue|book|make)",
]

_REFUSAL_PHRASES = [
    r"\brefuse\b",
    r"don'?t\s+want",
    r"not\s+giving",
    r"\bskip\b",
    r"\bdecline\b",
]

_APPOINTMENT_INTENT_PHRASES = [
    r"\bappointment\b",
    r"\bschedule\b",
    r"\bvisit\b",
    r"see\s+a\s+doctor",
    r"\bbook\b",
    r"make\s+an?\s+appointment",
    r"need\s+an?\s+appointment",
    r"need\s+to\s+come\s+in",
]

_compile = compile_phrases
_RE_WHY = _compile(*_WHY_PHRASES)
_RE_AFFIRMATION = _compile(*_AFFIRMATION_PHRASES)
_RE_DENIAL = _compile(*_DENIAL_PHRASES)
_RE_REFUSAL = _compile(*_REFUSAL_PHRASES)
_RE_APPOINTMENT_INTENT = _compile(*_APPOINTMENT_INTENT_PHRASES)

# Generic appointment-only transcript (no substantive reason given)
_RE_GENERIC_APPOINTMENT = re.compile(
    r"^(i\s+)?(need\s+(an?|to\s+make\s+(an?)?)|want\s+(an?|to\s+make\s+(an?)?)|"
    r"(would\s+like\s+(an?)?)|make\s+(an?)?)\s*appointment\s*\.?$",
    re.IGNORECASE,
)


# ── Intent enum ───────────────────────────────────────────────────────────────

class Intent:
    EMERGENCY = "emergency"
    HUMAN = "human"
    WHY = "why"
    AFFIRMATION = "affirmation"
    DENIAL = "denial"
    REFUSAL = "refusal"
    APPOINTMENT_INTENT = "appointment_intent"
    SUBSTANTIVE = "substantive"  # has real content, not merely one of the above


def classify_intent(transcript: str) -> str:
    """Return the dominant intent of a caller transcript.

    Order matters:
      emergency → human → why → refusal → denial → affirmation → appointment_intent → substantive
    Refusal precedes denial so "I don't want to give that" stays REFUSAL
    (identity-state routing needs that classification).
    Denial precedes affirmation so "not okay" doesn't match affirmation's
    bare \bokay\b token.
    """
    t = transcript.strip()
    if _RE_EMERGENCY.search(t):
        return Intent.EMERGENCY
    if _RE_HUMAN.search(t):
        return Intent.HUMAN
    if _RE_WHY.search(t):
        return Intent.WHY
    if _RE_REFUSAL.search(t):
        return Intent.REFUSAL
    if _RE_DENIAL.search(t):
        return Intent.DENIAL
    if _RE_AFFIRMATION.search(t):
        return Intent.AFFIRMATION
    if _RE_APPOINTMENT_INTENT.search(t):
        return Intent.APPOINTMENT_INTENT
    return Intent.SUBSTANTIVE


def is_generic_appointment_only(transcript: str) -> bool:
    """True when the transcript expresses appointment intent but no substantive reason."""
    return bool(_RE_GENERIC_APPOINTMENT.match(transcript.strip()))


# ── Pending patch dataclass ───────────────────────────────────────────────────

@dataclass
class _AppointmentPatch:
    """Fields to merge into AppointmentRequest when the proposal is accepted."""
    session_id: str
    created_at: datetime = field(default_factory=_utc_now)
    reason_for_visit: str | None = None
    patient_name: str | None = None
    date_of_birth: str | None = None
    insurance_name: str | None = None
    insurance_network_status: InsuranceNetworkStatus | None = None
    appointment_status: AppointmentStatus | None = None
    scheduled_slot: str | None = None

    def apply_to(self, existing: AppointmentRequest | None, appointment_id: str) -> AppointmentRequest:
        """Merge this patch onto an existing request, or create a new one."""
        base = existing or AppointmentRequest(
            appointment_id=appointment_id,
            session_id=self.session_id,
        )
        kwargs: dict[str, Any] = {}
        if self.reason_for_visit is not None:
            kwargs["reason_for_visit"] = self.reason_for_visit
        if self.patient_name is not None:
            kwargs["patient_name"] = self.patient_name
        if self.date_of_birth is not None:
            kwargs["date_of_birth"] = self.date_of_birth
        if self.insurance_name is not None:
            kwargs["insurance_name"] = self.insurance_name
        if self.insurance_network_status is not None:
            kwargs["insurance_network_status"] = self.insurance_network_status
        if self.appointment_status is not None:
            kwargs["appointment_status"] = self.appointment_status
        if self.scheduled_slot is not None:
            kwargs["scheduled_slot"] = self.scheduled_slot
        kwargs["updated_at"] = _utc_now()
        return replace(base, **kwargs)


# ── DemoRouter ────────────────────────────────────────────────────────────────

class DemoRouter:
    """Deterministic appointment-scheduling router for the browser demo.

    Constructed once during app lifespan.  Thread-safety: the pending-patch
    dict is mutated from async route handlers; in a single-process FastAPI
    deployment this is safe.  Multi-replica deployments would need an external
    store - that's out of scope for the demo.
    """

    def __init__(self, session_factory: sessionmaker, audit_secret: str) -> None:
        self._store = SQLAuditStore(session_factory, hmac_secret=audit_secret)
        self._pending: dict[str, _AppointmentPatch] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def propose(
        self,
        prompt: dict,
        source_turn_id: str,
        proposal_id: str | None = None,
    ) -> ModelProposal:
        """Classify intent, select the next template, and record a pending patch.

        The proposal_id passed in (or generated here) is used as both the
        ModelProposal.proposal_id AND the key in _pending.  /turn must call
        commit() or discard() with proposal.proposal_id.
        """
        self._prune_stale_patches()

        pid = proposal_id or _new_id()
        session_id: str = str(prompt.get("session_id", ""))
        current_state_raw: str = str(prompt.get("current_state", CallState.OPENING.value))
        # Most recent caller transcript (last entry in recent_turns where speaker=="caller")
        recent_turns: list[dict] = prompt.get("recent_turns", [])  # type: ignore[assignment]
        transcript = ""
        for turn in reversed(recent_turns):
            if turn.get("speaker") == "caller":
                transcript = str(turn.get("transcript", ""))
                break

        try:
            current_state = CallState(current_state_raw)
        except ValueError:
            current_state = CallState.OPENING

        existing = self._store.get_appointment_request(session_id) if session_id else None

        template_id, variables, transition, patch = self._dispatch(
            current_state=current_state,
            transcript=transcript,
            session_id=session_id,
            existing=existing,
        )

        if patch is not None:
            self._pending[pid] = patch

        return ModelProposal(
            proposal_id=pid,
            source_turn_id=source_turn_id,
            template_id=template_id,
            variables=variables,
            requested_transition=transition,
            model_version=_DEMO_MODEL_VERSION,
            session_id=session_id or None,
        )

    def commit(self, proposal_id: str) -> None:
        """Apply the pending patch for proposal_id to persistent storage."""
        patch = self._pending.pop(proposal_id, None)
        if patch is None:
            return
        session_id = patch.session_id
        existing = self._store.get_appointment_request(session_id)
        appointment_id = (existing.appointment_id if existing else _new_id())
        updated = patch.apply_to(existing, appointment_id)
        self._store.upsert_appointment_request(updated)

    def discard(self, proposal_id: str) -> None:
        """Drop the pending patch without writing to storage."""
        self._pending.pop(proposal_id, None)

    # ── Internal dispatch ─────────────────────────────────────────────────────

    def _dispatch(
        self,
        current_state: CallState,
        transcript: str,
        session_id: str,
        existing: AppointmentRequest | None,
    ) -> tuple[str, dict[str, str], CallState, _AppointmentPatch | None]:
        """Return (template_id, variables, next_state, patch_or_None)."""

        intent = classify_intent(transcript)

        # Emergency - highest priority, fires from any non-CLOSE state
        if intent == Intent.EMERGENCY:
            patch = _AppointmentPatch(
                session_id=session_id,
                appointment_status=AppointmentStatus.EMERGENCY_REDIRECT,
            )
            return "emergency_redirect", {}, CallState.EMERGENCY_EXIT, patch

        # Human handoff - second priority
        if intent == Intent.HUMAN:
            patch = _AppointmentPatch(
                session_id=session_id,
                appointment_status=AppointmentStatus.HUMAN_TAKEOVER,
            )
            return "human_handoff", {}, CallState.HUMAN_TAKEOVER, patch

        # Correction at IDENTITY/DEMOGRAPHICS/INSURANCE/READBACK → handoff to staff.
        # Demo router can't safely re-edit prior fields; staff handles it.
        # Intentionally NOT applied at OPENING or REASON_FOR_VISIT:
        #   - OPENING has nothing to correct yet
        #   - REASON_FOR_VISIT can innocuously contain "actually" / "wait" / "change"
        #     inside a valid medical reason ("change in vision", "waited too long")
        if current_state in {
            CallState.IDENTITY_CAPTURE,
            CallState.DEMOGRAPHICS,
            CallState.INSURANCE,
            CallState.READBACK,
        } and _is_correction(transcript):
            patch = _AppointmentPatch(
                session_id=session_id,
                appointment_status=AppointmentStatus.HUMAN_TAKEOVER,
            )
            return "human_handoff", {}, CallState.HUMAN_TAKEOVER, patch

        # State-specific dispatch
        if current_state == CallState.OPENING:
            return self._dispatch_opening(transcript, intent, session_id)

        if current_state == CallState.REASON_FOR_VISIT:
            return self._dispatch_reason(transcript, intent, session_id)

        if current_state == CallState.IDENTITY_CAPTURE:
            return self._dispatch_identity(transcript, intent, session_id)

        if current_state == CallState.DEMOGRAPHICS:
            return self._dispatch_demographics(transcript, intent, session_id)

        if current_state == CallState.INSURANCE:
            return self._dispatch_insurance(transcript, intent, session_id, existing)

        if current_state == CallState.READBACK:
            return self._dispatch_readback(intent, session_id, existing)

        # Fallback for any other state (DISPOSITION, CLOSE, etc.)
        return "hold_message", {}, current_state, None

    # ── Per-state handlers ────────────────────────────────────────────────────

    def _dispatch_opening(
        self, transcript: str, intent: str, session_id: str
    ) -> tuple[str, dict, CallState, _AppointmentPatch | None]:
        # At OPENING the assistant has not asked any question yet, so there's
        # no field-specific explanation that "why?" could refer to. Reply with
        # the same prompt we'd give for any opening interaction, including
        # "why?" - it's the most defensible behavior given no antecedent.
        # Note: emitting field_explanation_for_reason here would validator-reject
        # because that template's allowed_states is {REASON_FOR_VISIT} only.
        # Emergency / human handled before this dispatch.
        return "appointment_reason_prompt", {}, CallState.REASON_FOR_VISIT, None

    def _dispatch_reason(
        self, transcript: str, intent: str, session_id: str
    ) -> tuple[str, dict, CallState, _AppointmentPatch | None]:
        if intent == Intent.WHY:
            return "field_explanation_for_reason", {}, CallState.REASON_FOR_VISIT, None

        # Generic appointment-intent-only ("I need an appointment") - re-ask
        if intent == Intent.APPOINTMENT_INTENT and is_generic_appointment_only(transcript):
            return "appointment_reason_prompt", {}, CallState.REASON_FOR_VISIT, None

        # Substantive answer (or appointment intent with real content)
        if intent in (Intent.SUBSTANTIVE, Intent.APPOINTMENT_INTENT):
            patch = _AppointmentPatch(session_id=session_id, reason_for_visit=transcript.strip())
            return "patient_name_prompt", {}, CallState.IDENTITY_CAPTURE, patch

        # Affirmation / denial / refusal without context - re-ask
        return "appointment_reason_prompt", {}, CallState.REASON_FOR_VISIT, None

    def _dispatch_identity(
        self, transcript: str, intent: str, session_id: str
    ) -> tuple[str, dict, CallState, _AppointmentPatch | None]:
        if intent == Intent.WHY:
            return "field_explanation_for_name", {}, CallState.IDENTITY_CAPTURE, None

        # Refusal to give name - broader than the global REFUSAL intent at
        # this state: "none" / "no" / "n/a" alone all mean refusal here.
        if _is_identity_refusal(transcript, intent):
            patch = _AppointmentPatch(
                session_id=session_id,
                appointment_status=AppointmentStatus.HUMAN_TAKEOVER,
            )
            return "cannot_schedule_without_identity", {}, CallState.HUMAN_TAKEOVER, patch

        if intent in (Intent.SUBSTANTIVE, Intent.APPOINTMENT_INTENT):
            patch = _AppointmentPatch(session_id=session_id, patient_name=transcript.strip())
            return "dob_prompt", {}, CallState.DEMOGRAPHICS, patch

        # Affirmation / denial without content - re-ask
        return "patient_name_prompt", {}, CallState.IDENTITY_CAPTURE, None

    def _dispatch_demographics(
        self, transcript: str, intent: str, session_id: str
    ) -> tuple[str, dict, CallState, _AppointmentPatch | None]:
        if intent == Intent.WHY:
            return "field_explanation_for_dob", {}, CallState.DEMOGRAPHICS, None

        # Refusal to give DOB - broader than the global REFUSAL intent at
        # this state: "none" / "no" / "n/a" alone all mean refusal here.
        if _is_identity_refusal(transcript, intent):
            patch = _AppointmentPatch(
                session_id=session_id,
                appointment_status=AppointmentStatus.HUMAN_TAKEOVER,
            )
            return "cannot_schedule_without_identity", {}, CallState.HUMAN_TAKEOVER, patch

        # Try to parse a DOB from the transcript
        from .normalization import normalize_dob
        parsed = normalize_dob(transcript)
        if parsed is None:
            # No parseable DOB - ask for clarification (self-loop)
            return "dob_clarification", {}, CallState.DEMOGRAPHICS, None

        patch = _AppointmentPatch(session_id=session_id, date_of_birth=parsed)
        return "insurance_prompt", {}, CallState.INSURANCE, patch

    def _dispatch_insurance(
        self,
        transcript: str,
        intent: str,
        session_id: str,
        existing: AppointmentRequest | None,
    ) -> tuple[str, dict, CallState, _AppointmentPatch | None]:
        if intent == Intent.WHY:
            return "field_explanation_for_insurance", {}, CallState.INSURANCE, None

        # Off-script clinic question - interrogative starter only (NOT a bare "?").
        # "Aetna?" should still route through check_network_status; only phrases
        # like "Do you take walk-ins?" / "What about Medicare?" hand off.
        if _looks_like_question(transcript):
            patch = _AppointmentPatch(
                session_id=session_id,
                appointment_status=AppointmentStatus.HUMAN_TAKEOVER,
            )
            return "human_handoff", {}, CallState.HUMAN_TAKEOVER, patch

        # Explicit non-self-pay refusal ("I refuse to give insurance", "skip")
        if intent == Intent.REFUSAL and not _looks_like_self_pay(transcript):
            patch = _AppointmentPatch(
                session_id=session_id,
                appointment_status=AppointmentStatus.HUMAN_TAKEOVER,
            )
            return "cannot_schedule_without_identity", {}, CallState.HUMAN_TAKEOVER, patch

        # Run check_network_status on the transcript (handles self-pay internally)
        from .insurance import check_network_status
        status, display_name = check_network_status(transcript)

        if status == InsuranceNetworkStatus.NOT_PROVIDED:
            patch = _AppointmentPatch(
                session_id=session_id,
                insurance_name="self-pay",
                insurance_network_status=InsuranceNetworkStatus.NOT_PROVIDED,
            )
            return "insurance_not_provided", {}, CallState.READBACK, patch

        if status == InsuranceNetworkStatus.IN_NETWORK:
            patch = _AppointmentPatch(
                session_id=session_id,
                insurance_name=display_name,
                insurance_network_status=InsuranceNetworkStatus.IN_NETWORK,
            )
            return (
                "insurance_in_network",
                {"insurance_name": display_name},
                CallState.READBACK,
                patch,
            )

        if status == InsuranceNetworkStatus.OUT_OF_NETWORK:
            patch = _AppointmentPatch(
                session_id=session_id,
                insurance_name=display_name,
                insurance_network_status=InsuranceNetworkStatus.OUT_OF_NETWORK,
            )
            return (
                "insurance_out_of_network",
                {"insurance_name": display_name},
                CallState.READBACK,
                patch,
            )

        # UNKNOWN
        patch = _AppointmentPatch(
            session_id=session_id,
            insurance_name=display_name,
            insurance_network_status=InsuranceNetworkStatus.UNKNOWN,
        )
        return (
            "insurance_unknown",
            {"insurance_name": display_name},
            CallState.READBACK,
            patch,
        )

    def _dispatch_readback(
        self,
        intent: str,
        session_id: str,
        existing: AppointmentRequest | None,
    ) -> tuple[str, dict, CallState, _AppointmentPatch | None]:
        network_status = (
            existing.insurance_network_status
            if existing
            else InsuranceNetworkStatus.UNKNOWN
        )

        if intent == Intent.AFFIRMATION:
            scheduled_statuses = {
                InsuranceNetworkStatus.IN_NETWORK,
                InsuranceNetworkStatus.OUT_OF_NETWORK,
            }
            if network_status in scheduled_statuses:
                patch = _AppointmentPatch(
                    session_id=session_id,
                    appointment_status=AppointmentStatus.SCHEDULED,
                    scheduled_slot="next_available_demo_slot",
                )
                return "appointment_scheduled", {}, CallState.DISPOSITION, patch
            else:
                # UNKNOWN or NOT_PROVIDED - staff review
                patch = _AppointmentPatch(
                    session_id=session_id,
                    appointment_status=AppointmentStatus.STAFF_REVIEW,
                    scheduled_slot="next_available_demo_slot",
                )
                return "appointment_request_created", {}, CallState.DISPOSITION, patch

        # At READBACK, both DENIAL ("no", "not okay") and REFUSAL
        # ("don't want to schedule", "skip") mean negative - staff handoff.
        # Combining them ensures phrases like "I don't want to schedule"
        # (REFUSAL via "don't want") don't fall through to the re-ask path.
        if intent in (Intent.DENIAL, Intent.REFUSAL):
            patch = _AppointmentPatch(
                session_id=session_id,
                appointment_status=AppointmentStatus.STAFF_REVIEW,
            )
            return "human_handoff", {}, CallState.HUMAN_TAKEOVER, patch

        # Anything else - re-ask (self-loop); no patch.
        # Re-emit the right insurance confirmation template based on current network status.
        if network_status == InsuranceNetworkStatus.IN_NETWORK:
            ins_name = (existing.insurance_name or "") if existing else ""
            return (
                "insurance_in_network",
                {"insurance_name": ins_name},
                CallState.READBACK,
                None,
            )
        if network_status == InsuranceNetworkStatus.OUT_OF_NETWORK:
            ins_name = (existing.insurance_name or "") if existing else ""
            return (
                "insurance_out_of_network",
                {"insurance_name": ins_name},
                CallState.READBACK,
                None,
            )
        if network_status == InsuranceNetworkStatus.UNKNOWN:
            ins_name = (existing.insurance_name or "") if existing else ""
            return (
                "insurance_unknown",
                {"insurance_name": ins_name},
                CallState.READBACK,
                None,
            )
        return "insurance_not_provided", {}, CallState.READBACK, None

    # ── Pruning ───────────────────────────────────────────────────────────────

    def _prune_stale_patches(self) -> None:
        """Remove patches older than PATCH_TTL_SECONDS; hard-cap at PATCH_MAX_SIZE."""
        if len(self._pending) >= PATCH_MAX_SIZE:
            self._pending.clear()
            return
        cutoff = _utc_now().timestamp() - PATCH_TTL_SECONDS
        stale = [
            pid
            for pid, patch in self._pending.items()
            if patch.created_at.timestamp() < cutoff
        ]
        for pid in stale:
            del self._pending[pid]


# ── Helpers ───────────────────────────────────────────────────────────────────

_SELF_PAY_RE = re.compile(
    r"\b(none|self[\s\-]?pay|no\s+insurance|uninsured)\b", re.IGNORECASE
)


def _looks_like_self_pay(transcript: str) -> bool:
    return bool(_SELF_PAY_RE.search(transcript))


# Refusal-at-identity is broader than the global REFUSAL intent. At name/DOB
# the user saying just "none", "no", "n/a" means they're refusing to provide
# identifying information. INSURANCE state is intentionally NOT covered here -
# "none" at INSURANCE routes through check_network_status to NOT_PROVIDED.
_IDENTITY_REFUSAL_TOKENS = {"none", "n/a", "na", "no"}


def _is_identity_refusal(transcript: str, intent: str) -> bool:
    value = transcript.strip().lower()
    return intent == Intent.REFUSAL or value in _IDENTITY_REFUSAL_TOKENS


# Correction language at IDENTITY/DEMOGRAPHICS/INSURANCE/READBACK → handoff.
# Caller said "actually...", "wait...", "wrong", "change..." - demo router
# cannot safely re-edit prior fields, so staff handles it.
_CORRECTION_PHRASES = [
    r"\bactually\b",
    r"\bwait\b",
    r"\bcorrection\b",
    r"\bwrong\b",
    r"\bnot right\b",
    r"\bchange\b",
]
_RE_CORRECTION = _compile(*_CORRECTION_PHRASES)


def _is_correction(transcript: str) -> bool:
    return bool(_RE_CORRECTION.search(transcript))


# Off-script clinic question detector for INSURANCE state.
# Keys ONLY on interrogative starters (do/does/are/is/can/will/should/how/
# when/where/what/why) - NOT a bare "?". This way "Aetna?" still routes
# through check_network_status (punctuation is stripped by the normalizer)
# and "Do you take walk-ins?" hands off to staff.
_QUESTION_RE = re.compile(
    r"^(do|does|are|is|can|will|should|how|when|where|what|why)\b",
    re.IGNORECASE,
)


def _looks_like_question(transcript: str) -> bool:
    return bool(_QUESTION_RE.search(transcript.strip()))
