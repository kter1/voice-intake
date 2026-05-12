from __future__ import annotations

from uuid import uuid4

from voice_intake.intent_patterns import (
    RE_EMERGENCY,
    RE_HUMAN,
    RE_OUT_OF_SCOPE,
    has_prompt_injection,
)
from voice_intake.models import CallState, ModelProposal

_MODEL_VERSION = "safety-prerouter-v1"

_HANDOFF_STATES = {
    CallState.OPENING,
    CallState.REASON_FOR_VISIT,
    CallState.IDENTITY_CAPTURE,
    CallState.DEMOGRAPHICS,
    CallState.INSURANCE,
    CallState.READBACK,
    CallState.HUMAN_TAKEOVER,
}

_EARLY_OUT_OF_SCOPE_STATES = {
    CallState.OPENING,
    CallState.REASON_FOR_VISIT,
}


class SafetyPreRouter:
    """Deterministic safety and scope routing before model/router dispatch."""

    def check(
        self,
        transcript: str,
        current_state: CallState,
        source_turn_id: str,
        proposal_id: str | None = None,
        session_id: str | None = None,
    ) -> ModelProposal | None:
        text = transcript.strip()
        if not text:
            return None

        pid = proposal_id or str(uuid4())

        if current_state != CallState.CLOSE and RE_EMERGENCY.search(text):
            return ModelProposal(
                proposal_id=pid,
                source_turn_id=source_turn_id,
                template_id="emergency_redirect",
                variables={},
                requested_transition=CallState.EMERGENCY_EXIT,
                model_version=_MODEL_VERSION,
                session_id=session_id,
            )

        if current_state in _HANDOFF_STATES and RE_HUMAN.search(text):
            return self._handoff(pid, source_turn_id, session_id)

        if current_state in _HANDOFF_STATES and has_prompt_injection(text):
            return self._handoff(pid, source_turn_id, session_id)

        if current_state in _EARLY_OUT_OF_SCOPE_STATES and RE_OUT_OF_SCOPE.search(text):
            return self._handoff(pid, source_turn_id, session_id)

        return None

    @staticmethod
    def _handoff(
        proposal_id: str,
        source_turn_id: str,
        session_id: str | None,
    ) -> ModelProposal:
        return ModelProposal(
            proposal_id=proposal_id,
            source_turn_id=source_turn_id,
            template_id="human_handoff",
            variables={},
            requested_transition=CallState.HUMAN_TAKEOVER,
            model_version=_MODEL_VERSION,
            session_id=session_id,
        )
