from __future__ import annotations

from voice_intake.config import Settings
from voice_intake.models import CallState, ModelProposal
from voice_intake.testing.mock_llm import MockLLM


def _current_state(prompt: dict) -> CallState:
    try:
        return CallState(str(prompt.get("current_state", CallState.OPENING.value)))
    except ValueError:
        return CallState.OPENING


def _latest_caller_transcript(prompt: dict) -> str:
    recent_turns = prompt.get("recent_turns") or []
    if isinstance(recent_turns, list):
        for turn in reversed(recent_turns):
            if isinstance(turn, dict) and turn.get("speaker") == "caller":
                return str(turn.get("transcript", ""))
    return str(prompt.get("transcript", ""))


async def propose_next_action(
    prompt: dict,
    source_turn_id: str,
    settings: Settings,
    llm_client: object | None,
    demo_router: object | None = None,
    safety_pre_router: object | None = None,
    proposal_id: str | None = None,
) -> ModelProposal | None:
    if safety_pre_router is not None:
        proposal = safety_pre_router.check(
            transcript=_latest_caller_transcript(prompt),
            current_state=_current_state(prompt),
            source_turn_id=source_turn_id,
            proposal_id=proposal_id,
            session_id=str(prompt.get("session_id") or "") or None,
        )
        if proposal is not None:
            return proposal

    # DemoRouter takes priority over everything else when the flag is set.
    if settings.demo_router:
        if demo_router is None:
            return None
        return demo_router.propose(prompt, source_turn_id, proposal_id=proposal_id)
    if settings.mock_llm:
        return MockLLM().propose(prompt, source_turn_id)
    if llm_client is not None:
        return await llm_client.propose(
            prompt, source_turn_id=source_turn_id, proposal_id=proposal_id
        )
    return None
