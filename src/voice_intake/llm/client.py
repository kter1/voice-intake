from __future__ import annotations

import json
import logging
import os
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


class LLMClient:
    """Generic remote AI client using the chat-completions protocol.
    Works with OpenAI-compatible chat-completions endpoints.
    Returns None on unrecoverable error so the caller can trigger takeover logic.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 30.0,
        prompt_profile: PromptProfile | str | None = None,
        chat_completions_path: str = "/v1/chat/completions",
        max_retries: int = 3,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=timeout)
        self._model = model
        self._prompt_profile = prompt_profile or os.getenv("LLM_PROMPT_PROFILE", "default")
        self._chat_completions_path = chat_completions_path
        self._max_retries = max_retries

    async def propose(
        self,
        prompt: dict,
        source_turn_id: str,
        proposal_id: str | None = None,
    ) -> ModelProposal | None:
        proposal_id = proposal_id or str(uuid4())
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": build_system_prompt(
                        prompt,
                        prompt_profile=self._prompt_profile,
                    ),
                },
                {"role": "user", "content": build_user_message(prompt)},
            ],
            "tools": [OAI_PROPOSE_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "propose_action"}},
        }

        async def _call() -> ModelProposal:
            response = await self._client.post(self._chat_completions_path, json=payload)
            response.raise_for_status()
            body = response.json()
            args = json.loads(
                body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
            )
            return ModelProposal(
                proposal_id=proposal_id,
                source_turn_id=source_turn_id,
                template_id=args["template_id"],
                variables=args.get("variables", {}),
                requested_transition=coerce_transition(args.get("requested_transition")),
                model_version=self._model,
                spoken_text=str(args["spoken_response"]) if args.get("spoken_response") else None,
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
            _log.warning("LLMClient.propose failed: %s", exc)
            return None

    async def aclose(self) -> None:
        await self._client.aclose()
