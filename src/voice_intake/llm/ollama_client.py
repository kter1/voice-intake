from __future__ import annotations

import json
import logging
import os
import re
from uuid import uuid4

import httpx

from voice_intake.models import ModelProposal

from .common import (
    OAI_PROPOSE_TOOL,
    PromptProfile,
    build_system_prompt,
    build_user_message,
    coerce_transition,
)
from .retry import with_exponential_backoff

_log = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?[ \t]*\n(.*?)\n```", re.DOTALL)

# Appended to the Ollama system prompt only - remote LLMClient uses plain tool-use.
_OLLAMA_JSON_FALLBACK_INSTRUCTION = """\
If tool calling is unavailable, output ONLY a single compact JSON object. No prose.

JSON object must have exactly these three keys:
- "template_id": one of the available_templates from the prompt
- "variables": an object (usually {} unless the template requires fields)
- "requested_transition": when the prompt includes an `allowed_transitions` list,
  copy this value EXACTLY from that list. When no list is present, choose from
  these CallState values:
  opening, consent_recording, consent_ai_assistance, consent_declined,
  identity_capture, demographics, insurance, reason_for_visit, readback,
  disposition, close, human_takeover, emergency_exit, manual_mode

CRITICAL: requested_transition is NOT a template ID. It is a state name.
Never put a template_id value into requested_transition.

CRITICAL: do not request consent_recording, consent_ai_assistance,
consent_declined, or manual_mode unless they appear in `allowed_transitions`.

Example:
{"template_id":"appointment_reason_prompt","variables":{},"requested_transition":"reason_for_visit"}

Output the JSON object only. No markdown fences, no prose, no explanation."""


def _extract_json_args(content: str | None) -> dict | None:
    """Try to parse propose_action args from message.content.

    Accepts plain compact JSON or markdown-fenced JSON (```json ... ```).
    Returns None for prose, malformed JSON, or dicts missing template_id.
    """
    if not content:
        return None
    text = content.strip()
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict) or "template_id" not in parsed:
        return None
    return parsed


def _build_template_constraint(
    available_templates: list[str] | None,
    allowed_transitions: list[str] | None = None,
) -> str:
    """Build a per-turn system block that lists allowed template IDs and transitions.

    Keeps local to OllamaClient; remote LLMClient uses native tool-use schema.
    Returns empty string when no templates are provided.
    """
    if not available_templates:
        return ""
    parts = []
    joined = ", ".join(available_templates)
    parts.append(
        f"\nFor this turn, valid template_id values are EXACTLY: {joined}\n"
        "Copy template_id exactly from this list. Do not invent template_id values.\n"
        "Do not output identity_prompt, introduction_prompt, answer_question, who_are_you, "
        "or any other name not in this list.\n"
        "If asked 'Who are you?', still choose an existing valid template_id from the list."
    )
    if allowed_transitions:
        joined_t = ", ".join(allowed_transitions)
        parts.append(
            f"\nFor this turn, valid requested_transition values are EXACTLY: {joined_t}\n"
            "Copy requested_transition exactly from this list. Do not invent transitions.\n"
            "Do not request any state not in this list."
        )
    return "".join(parts)


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 30.0,
        prompt_profile: PromptProfile | str | None = None,
        max_retries: int = 3,
    ) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)
        self._model = model
        self._prompt_profile = prompt_profile or os.getenv("LLM_PROMPT_PROFILE", "default")
        self._max_retries = max_retries

    async def aclose(self) -> None:
        await self._client.aclose()

    async def propose(
        self,
        prompt: dict,
        source_turn_id: str,
        proposal_id: str | None = None,
    ) -> ModelProposal | None:
        proposal_id = proposal_id or str(uuid4())
        system_content = (
            build_system_prompt(prompt, prompt_profile=self._prompt_profile)
            + "\n" + _OLLAMA_JSON_FALLBACK_INSTRUCTION
            + _build_template_constraint(
                prompt.get("available_templates"),
                prompt.get("allowed_transitions"),
            )
        )
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": build_user_message(prompt)},
            ],
            "tools": [OAI_PROPOSE_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "propose_action"}},
        }

        async def _call() -> ModelProposal:
            response = await self._client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            body = response.json()
            message = body["choices"][0]["message"]
            tool_calls = message.get("tool_calls")
            if tool_calls:
                args = json.loads(tool_calls[0]["function"]["arguments"])
            else:
                args = _extract_json_args(message.get("content"))
                if args is None:
                    raise KeyError("tool_calls")
            return ModelProposal(
                proposal_id=proposal_id,
                source_turn_id=source_turn_id,
                template_id=args["template_id"],
                variables=args.get("variables", {}),
                requested_transition=coerce_transition(args.get("requested_transition")),
                model_version=self._model,
            )

        try:
            return await with_exponential_backoff(
                _call, max_retries=self._max_retries, base_delay=0.5
            )
        except (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.HTTPStatusError,
            KeyError,
            IndexError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            _log.warning("OllamaClient.propose failed: %s", exc)
            return None
