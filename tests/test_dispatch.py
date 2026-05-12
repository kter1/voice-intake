"""
Unit tests for voice_intake.llm.dispatch.propose_next_action - confirms
demo_router takes priority over mock_llm and remote LLM clients when the
flag is set.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from voice_intake.config import Settings
from voice_intake.llm.dispatch import propose_next_action
from voice_intake.models import CallState, ModelProposal


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _fake_proposal(source_turn_id: str, proposal_id: str | None = None) -> ModelProposal:
    return ModelProposal(
        proposal_id=proposal_id or "test-pid",
        source_turn_id=source_turn_id,
        template_id="appointment_reason_prompt",
        variables={},
        requested_transition=CallState.REASON_FOR_VISIT,
        model_version="fake-v1",
    )


def test_demo_router_takes_priority_over_mock_llm_when_flag_set():
    settings = Settings(demo_router=True, mock_llm=True)
    demo = MagicMock()
    demo.propose.return_value = _fake_proposal("turn-1", "pid-1")

    result = _run(propose_next_action(
        prompt={"current_state": "opening", "session_id": "s-1"},
        source_turn_id="turn-1",
        settings=settings,
        llm_client=None,
        demo_router=demo,
        proposal_id="pid-1",
    ))

    demo.propose.assert_called_once_with(
        {"current_state": "opening", "session_id": "s-1"},
        "turn-1",
        proposal_id="pid-1",
    )
    assert result is not None
    assert result.proposal_id == "pid-1"


def test_safety_pre_router_takes_priority_over_demo_router():
    settings = Settings(demo_router=True, mock_llm=True)
    demo = MagicMock()
    safety = MagicMock()
    safety.check.return_value = ModelProposal(
        proposal_id="pid-safe",
        source_turn_id="turn-1",
        template_id="emergency_redirect",
        variables={},
        requested_transition=CallState.EMERGENCY_EXIT,
        model_version="safety-prerouter-v1",
        session_id="s-1",
    )

    result = _run(propose_next_action(
        prompt={
            "current_state": "opening",
            "session_id": "s-1",
            "recent_turns": [{"speaker": "caller", "transcript": "Bleeding"}],
        },
        source_turn_id="turn-1",
        settings=settings,
        llm_client=None,
        demo_router=demo,
        safety_pre_router=safety,
        proposal_id="pid-safe",
    ))

    safety.check.assert_called_once_with(
        transcript="Bleeding",
        current_state=CallState.OPENING,
        source_turn_id="turn-1",
        proposal_id="pid-safe",
        session_id="s-1",
    )
    demo.propose.assert_not_called()
    assert result is not None
    assert result.template_id == "emergency_redirect"


def test_returns_none_when_demo_router_flag_set_but_router_missing():
    """Defensive: DEMO_ROUTER=true but no router instance wired into deps."""
    settings = Settings(demo_router=True, mock_llm=True)

    result = _run(propose_next_action(
        prompt={"current_state": "opening", "session_id": "s-1"},
        source_turn_id="turn-1",
        settings=settings,
        llm_client=None,
        demo_router=None,
        proposal_id="pid-1",
    ))

    assert result is None


def test_mock_llm_used_when_demo_router_disabled():
    settings = Settings(demo_router=False, mock_llm=True)

    result = _run(propose_next_action(
        prompt={"current_state": "opening", "session_id": "s-1"},
        source_turn_id="turn-1",
        settings=settings,
        llm_client=None,
        demo_router=None,
    ))

    # MockLLM returns a real ModelProposal for every state.
    assert result is not None
    assert result.source_turn_id == "turn-1"


def test_remote_llm_used_when_no_mocks_or_demo():
    """When all mock/demo flags are False, the remote llm_client is invoked."""
    settings = Settings(demo_router=False, mock_llm=False)

    class _FakeRemoteLLM:
        def __init__(self):
            self.calls = []

        async def propose(self, prompt, source_turn_id, proposal_id=None):
            self.calls.append((prompt, source_turn_id, proposal_id))
            return _fake_proposal(source_turn_id, proposal_id)

    fake = _FakeRemoteLLM()
    result = _run(propose_next_action(
        prompt={"current_state": "opening", "session_id": "s-1"},
        source_turn_id="turn-1",
        settings=settings,
        llm_client=fake,
        demo_router=None,
        proposal_id="pid-remote",
    ))

    assert result is not None
    assert result.proposal_id == "pid-remote"
    assert len(fake.calls) == 1


def test_demo_router_gets_proposal_id_passed_through():
    """proposal_id round-trip: turns.py → propose_next_action → DemoRouter.propose."""
    settings = Settings(demo_router=True)
    demo = MagicMock()
    demo.propose.return_value = _fake_proposal("turn-1", "round-trip-id")

    _run(propose_next_action(
        prompt={"current_state": "opening", "session_id": "s-1"},
        source_turn_id="turn-1",
        settings=settings,
        llm_client=None,
        demo_router=demo,
        proposal_id="round-trip-id",
    ))

    # The same proposal_id passed in must reach DemoRouter.propose.
    _, kwargs = demo.propose.call_args
    assert kwargs["proposal_id"] == "round-trip-id"
