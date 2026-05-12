from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from voice_intake.llm.client import LLMClient
from voice_intake.llm.common import build_system_prompt


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _tool_response() -> MagicMock:
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": json.dumps(
                                    {
                                        "template_id": "appointment_reason_prompt",
                                        "variables": {},
                                        "requested_transition": "reason_for_visit",
                                    }
                                )
                            }
                        }
                    ]
                }
            }
        ]
    }
    return response


def test_default_prompt_profile_preserves_current_system_prompt():
    prompt = build_system_prompt({"current_state": "opening"})

    assert "You are a supervised AI voice intake assistant" in prompt
    assert "Recruiter demo profile" not in prompt
    assert "appointment scheduling intake only" not in prompt


def test_recruiter_prompt_profile_adds_appointment_only_boundaries():
    prompt = build_system_prompt(
        {"current_state": "opening"},
        prompt_profile="recruiter_demo",
    )

    assert "Recruiter demo profile" in prompt
    assert "appointment scheduling intake only" in prompt
    assert "Your ONLY output is a call to the `propose_action` tool" in prompt
    assert "Do not invent fields" in prompt
    assert "coverage guarantees" in prompt
    assert "Emergency or safety language" in prompt


def test_demo_prompt_profile_alias_matches_recruiter_demo_profile():
    prompt = {"current_state": "opening"}

    assert build_system_prompt(prompt, prompt_profile="demo") == build_system_prompt(
        prompt,
        prompt_profile="recruiter_demo",
    )


def test_settings_accepts_demo_prompt_profile_alias():
    from voice_intake.config import Settings

    settings = Settings(llm_prompt_profile="demo")

    assert settings.llm_prompt_profile == "demo"


def test_remote_client_uses_recruiter_prompt_profile():
    client = LLMClient(
        base_url="https://llm.example.test",
        model="demo-model",
        timeout=1.0,
        prompt_profile="recruiter_demo",
    )
    client._client.post = AsyncMock(return_value=_tool_response())

    proposal = _run(client.propose({"current_state": "opening"}, "turn-1", "proposal-1"))

    assert proposal is not None
    payload = client._client.post.call_args.kwargs["json"]
    system_message = payload["messages"][0]["content"]
    assert "Recruiter demo profile" in system_message
    assert "appointment scheduling intake only" in system_message
    _run(client.aclose())


def test_recruiter_docs_section_avoids_forbidden_claims():
    readme = open("README.md", encoding="utf-8").read()
    section = readme.split("## Quickstart", 1)[1]
    section = section.split("## What's implemented", 1)[0].lower()

    forbidden = [
        "hipaa",
        "baa",
        "zdr",
        "jwt",
        "jwks",
        "certified",
        "production-ready",
        "production readiness",
    ]
    assert not any(term in section for term in forbidden)
