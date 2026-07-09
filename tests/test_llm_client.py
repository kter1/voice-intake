"""
Unit tests for the LLM integration layer.
All tests mock httpx - no real API calls are made.
"""
import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock

import httpx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_oai_response(template_id: str, transition: str, variables: dict | None = None,
                       model: str = "test-model"):
    """Build a mock httpx Response shaped like a chat-completions tool-call reply."""
    body = {
        "choices": [{
            "message": {
                "tool_calls": [{
                    "function": {
                        "arguments": json.dumps({
                            "template_id": template_id,
                            "variables": variables or {},
                            "requested_transition": transition,
                        })
                    }
                }]
            }
        }]
    }
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = body
    return response


# ---------------------------------------------------------------------------
# CoerceTransitionTests
# ---------------------------------------------------------------------------

class CoerceTransitionTest(unittest.TestCase):
    def test_none_raises_value_error(self):
        from voice_intake.llm.common import coerce_transition
        with self.assertRaises(ValueError):
            coerce_transition(None)

    def test_empty_string_raises_value_error(self):
        from voice_intake.llm.common import coerce_transition
        with self.assertRaises(ValueError):
            coerce_transition("")

    def test_invalid_state_name_raises_value_error(self):
        from voice_intake.llm.common import coerce_transition
        with self.assertRaises(ValueError):
            coerce_transition("not_a_real_state")

    def test_valid_state_string_returns_enum(self):
        from voice_intake.llm.common import coerce_transition
        from voice_intake.models import CallState
        result = coerce_transition("identity_capture")
        self.assertEqual(result, CallState.IDENTITY_CAPTURE)

    def test_callstate_enum_passthrough(self):
        from voice_intake.llm.common import coerce_transition
        from voice_intake.models import CallState
        result = coerce_transition(CallState.IDENTITY_CAPTURE)
        self.assertEqual(result, CallState.IDENTITY_CAPTURE)


# ---------------------------------------------------------------------------
# RetryTests
# ---------------------------------------------------------------------------

class RetryTest(unittest.TestCase):
    def test_succeeds_on_first_try(self):
        from voice_intake.llm.retry import with_exponential_backoff

        called = []

        async def factory():
            called.append(1)
            return "ok"

        result = _run(with_exponential_backoff(factory, max_retries=3, base_delay=0.0))
        self.assertEqual(result, "ok")
        self.assertEqual(len(called), 1)

    def test_retries_on_timeout_error(self):
        from voice_intake.llm.retry import with_exponential_backoff

        calls = []

        async def factory():
            calls.append(1)
            if len(calls) < 3:
                raise asyncio.TimeoutError()
            return "done"

        result = _run(with_exponential_backoff(factory, max_retries=3, base_delay=0.0))
        self.assertEqual(result, "done")
        self.assertEqual(len(calls), 3)

    def test_raises_after_max_retries_exhausted(self):
        from voice_intake.llm.retry import with_exponential_backoff

        async def factory():
            raise asyncio.TimeoutError()

        with self.assertRaises(asyncio.TimeoutError):
            _run(with_exponential_backoff(factory, max_retries=2, base_delay=0.0))

    def test_non_retryable_error_raises_immediately(self):
        from voice_intake.llm.retry import with_exponential_backoff

        calls = []

        async def factory():
            calls.append(1)
            raise ValueError("non-retryable")

        with self.assertRaises(ValueError):
            _run(with_exponential_backoff(factory, max_retries=3, base_delay=0.0))
        self.assertEqual(len(calls), 1)

    def test_httpx_connect_error_is_retryable(self):
        from voice_intake.llm.retry import _is_retryable

        error = httpx.ConnectError("offline", request=httpx.Request("POST", "http://localhost"))
        self.assertTrue(_is_retryable(error))

    def test_httpx_429_is_retryable(self):
        from voice_intake.llm.retry import _is_retryable

        response = httpx.Response(429, request=httpx.Request("POST", "http://localhost"))
        error = httpx.HTTPStatusError("rate limited", request=response.request, response=response)
        self.assertTrue(_is_retryable(error))


# ---------------------------------------------------------------------------
# LLMClientTests - generic remote AI client
# ---------------------------------------------------------------------------

class LLMClientTest(unittest.TestCase):

    def _make_client(self):
        from voice_intake.llm.client import LLMClient
        return LLMClient(base_url="http://ai.example.com/v1", model="test-model", timeout=5.0)

    def test_propose_returns_model_proposal_on_success(self):
        from voice_intake.llm.client import LLMClient
        from voice_intake.models import CallState

        client = self._make_client()
        client._client.post = AsyncMock(return_value=_make_oai_response(
            "collect_field_prompt", "identity_capture"
        ))
        prompt = {
            "current_state": "identity_capture",
            "available_templates": ["collect_field_prompt"],
            "slot_status": [],
            "recent_turns": [],
        }
        proposal = _run(client.propose(prompt, "turn-1", "prop-1"))
        _run(client.aclose())

        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.template_id, "collect_field_prompt")
        self.assertEqual(proposal.requested_transition, CallState.IDENTITY_CAPTURE)
        self.assertEqual(proposal.proposal_id, "prop-1")
        self.assertEqual(proposal.source_turn_id, "turn-1")
        self.assertEqual(proposal.model_version, "test-model")

    def test_propose_returns_none_on_connect_error(self):
        from voice_intake.llm.client import LLMClient

        client = self._make_client()
        client._client.post = AsyncMock(
            side_effect=httpx.ConnectError("down", request=httpx.Request("POST", "http://ai.example.com"))
        )
        result = _run(client.propose({}, "turn-1"))
        _run(client.aclose())

        self.assertIsNone(result)

    def test_propose_returns_none_on_timeout(self):
        from voice_intake.llm.client import LLMClient

        client = self._make_client()
        client._client.post = AsyncMock(
            side_effect=httpx.TimeoutException("timeout")
        )
        result = _run(client.propose({}, "turn-1"))
        _run(client.aclose())

        self.assertIsNone(result)

    def test_propose_returns_none_on_missing_tool_calls(self):
        from voice_intake.llm.client import LLMClient

        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {"choices": [{"message": {}}]}

        client = self._make_client()
        client._client.post = AsyncMock(return_value=response)
        result = _run(client.propose({}, "turn-1"))
        _run(client.aclose())

        self.assertIsNone(result)

    def test_propose_returns_none_on_empty_requested_transition(self):
        from voice_intake.llm.client import LLMClient

        client = self._make_client()
        client._client.post = AsyncMock(return_value=_make_oai_response(
            "collect_field_prompt", ""
        ))
        result = _run(client.propose({}, "turn-1"))
        _run(client.aclose())

        self.assertIsNone(result)

    def test_propose_returns_none_on_missing_requested_transition(self):
        from voice_intake.llm.client import LLMClient

        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"tool_calls": [{"function": {"arguments": json.dumps({
                "template_id": "collect_field_prompt",
                "variables": {},
                # requested_transition deliberately omitted
            })}}]}}]
        }
        client = self._make_client()
        client._client.post = AsyncMock(return_value=response)
        result = _run(client.propose({}, "turn-1"))
        _run(client.aclose())

        self.assertIsNone(result)

    def test_propose_returns_none_on_malformed_json_arguments(self):
        from voice_intake.llm.client import LLMClient

        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "choices": [{
                "message": {
                    "tool_calls": [{"function": {"arguments": "not-valid-json{"}}]
                }
            }]
        }

        client = self._make_client()
        client._client.post = AsyncMock(return_value=response)
        result = _run(client.propose({}, "turn-1"))
        _run(client.aclose())

        self.assertIsNone(result)

    def test_knowledge_context_injected_into_system(self):
        """RAG chunks must appear in the system message, not the user message."""
        from voice_intake.llm.common import build_system_prompt

        prompt = {
            "current_state": "identity_capture",
            "knowledge_context": [
                {"text": "Appointments are 30 minutes.", "source": "s1",
                 "category": "policy", "score": 0.9}
            ],
        }
        system = build_system_prompt(prompt)
        self.assertIn("[CONTEXT]", system)
        self.assertIn("Appointments are 30 minutes.", system)

    def test_patient_context_injected(self):
        from voice_intake.llm.common import build_system_prompt

        prompt = {
            "patient_context": {"first_name": "Jane", "dob": "****-03-14"},
        }
        system = build_system_prompt(prompt)
        self.assertIn("[PATIENT_CONTEXT]", system)
        self.assertIn("Jane", system)

    def test_eligibility_context_injected(self):
        from voice_intake.llm.common import build_system_prompt

        prompt = {
            "eligibility_context": {"eligible": True, "plan_name": "BlueCross"},
        }
        system = build_system_prompt(prompt)
        self.assertIn("[ELIGIBILITY]", system)

    def test_prior_calls_context_injected(self):
        from voice_intake.llm.common import build_system_prompt

        prompt = {
            "prior_calls_context": [{"session_id": "old-1", "disposition": "intake_complete"}],
        }
        system = build_system_prompt(prompt)
        self.assertIn("[PRIOR_CONTEXT]", system)

    def test_rag_keys_stripped_from_user_message(self):
        """RAG keys must not appear in the JSON sent as the user message."""
        from voice_intake.llm.common import build_user_message

        prompt = {
            "current_state": "identity_capture",
            "knowledge_context": [{"text": "policy text"}],
            "patient_context": {"first_name": "Jane"},
        }
        user_content = build_user_message(prompt)
        user_dict = json.loads(user_content)
        self.assertNotIn("knowledge_context", user_dict)
        self.assertNotIn("patient_context", user_dict)
        self.assertIn("current_state", user_dict)

    def test_system_prompt_mentions_allowed_transitions_authoritative(self):
        """C4.4: system prompt instructs the model to copy from allowed_transitions."""
        from voice_intake.llm.common import build_system_prompt
        system = build_system_prompt({})
        self.assertIn("allowed_transitions", system)
        self.assertIn("AUTHORITATIVE", system)

    def test_allowed_transitions_passes_through_to_user_message(self):
        """C4.4: allowed_transitions reaches the user JSON unmodified."""
        from voice_intake.llm.common import build_user_message
        prompt = {
            "current_state": "opening",
            "allowed_transitions": ["opening", "reason_for_visit"],
        }
        user_content = build_user_message(prompt)
        user_dict = json.loads(user_content)
        self.assertEqual(user_dict["allowed_transitions"], ["opening", "reason_for_visit"])

    def test_tool_schema_documents_allowed_transitions_constraint(self):
        """C4.4: requested_transition description references allowed_transitions."""
        from voice_intake.llm.common import OAI_PROPOSE_TOOL
        desc = OAI_PROPOSE_TOOL["function"]["parameters"]["properties"]["requested_transition"]["description"]
        self.assertIn("allowed_transitions", desc)


# ---------------------------------------------------------------------------
# DispatchTests
# ---------------------------------------------------------------------------

class DispatchTest(unittest.TestCase):

    def _make_settings(self, mock_llm: bool = True):
        from voice_intake.config import Settings
        return Settings(mock_llm=mock_llm)

    def test_dispatch_uses_mock_llm_when_flag_set(self):
        from voice_intake.llm.dispatch import propose_next_action

        settings = self._make_settings(mock_llm=True)
        prompt = {
            "current_state": "identity_capture",
            "available_templates": ["collect_field_prompt"],
            "slot_status": [],
            "recent_turns": [],
        }
        result = _run(propose_next_action(prompt, "turn-1", settings, llm_client=None))
        self.assertIsNotNone(result)

    def test_dispatch_calls_shared_client_when_provided(self):
        from voice_intake.llm.dispatch import propose_next_action

        settings = self._make_settings(mock_llm=False)
        mock_client = MagicMock()
        mock_client.propose = AsyncMock(return_value=MagicMock())

        _run(propose_next_action({}, "turn-1", settings, llm_client=mock_client))
        mock_client.propose.assert_called_once()

    def test_dispatch_returns_none_when_no_client(self):
        from voice_intake.llm.dispatch import propose_next_action

        settings = self._make_settings(mock_llm=False)
        result = _run(propose_next_action({}, "turn-1", settings, llm_client=None))
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# LLMClient Chat-Completions Path Tests
# ---------------------------------------------------------------------------

class LLMClientChatCompletionsPathTest(unittest.TestCase):
    """Test configurable chat-completions path for Gemini/OpenAI-compatible providers."""

    def test_default_path_is_v1_chat_completions(self):
        """Settings default preserves OpenAI/Groq behavior."""
        from voice_intake.config import Settings
        settings = Settings()
        self.assertEqual(settings.llm_chat_completions_path, "/v1/chat/completions")

    def test_llm_client_default_path(self):
        """LLMClient defaults to /v1/chat/completions when not provided."""
        from voice_intake.llm.client import LLMClient

        client = LLMClient(
            base_url="https://api.example.com",
            model="test-model",
            api_key="test-key",
        )
        self.assertEqual(client._chat_completions_path, "/v1/chat/completions")

    def test_llm_client_custom_path_gemini(self):
        """LLMClient uses custom path when provided (Gemini example)."""
        from voice_intake.llm.client import LLMClient

        client = LLMClient(
            base_url="https://api.example.com",
            model="gemini-2.5-flash-lite",
            api_key="test-key",
            chat_completions_path="/chat/completions",
        )
        self.assertEqual(client._chat_completions_path, "/chat/completions")

    def test_llm_client_uses_custom_path_in_post(self):
        """LLMClient.propose() uses the configured path."""
        from voice_intake.llm.client import LLMClient
        from unittest.mock import patch, AsyncMock

        client = LLMClient(
            base_url="https://api.example.com",
            model="gemini-2.5-flash-lite",
            api_key="test-key",
            chat_completions_path="/chat/completions",
        )

        # Mock the httpx client's post method
        mock_response = _make_oai_response("opening_disclosure", "opening")
        with patch.object(client._client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            prompt = {
                "session_id": "s-123",
                "current_state": "opening",
                "transcript": "Who are you?",
                "recent_turns": [],
            }

            _run(client.propose(prompt, "turn-1"))

            # Verify post was called with the custom path
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            self.assertEqual(call_args[0][0], "/chat/completions")

    def test_llm_client_default_path_in_post(self):
        """LLMClient.propose() uses /v1/chat/completions by default."""
        from voice_intake.llm.client import LLMClient
        from unittest.mock import patch, AsyncMock

        client = LLMClient(
            base_url="https://api.example.com",
            model="test-model",
            api_key="test-key",
        )

        mock_response = _make_oai_response("opening_disclosure", "opening")
        with patch.object(client._client, 'post', new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            prompt = {
                "session_id": "s-123",
                "current_state": "opening",
                "transcript": "Who are you?",
                "recent_turns": [],
            }

            _run(client.propose(prompt, "turn-1"))

            # Verify post was called with the default path
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            self.assertEqual(call_args[0][0], "/v1/chat/completions")

    def test_build_llm_client_passes_path_from_settings(self):
        """_build_llm_client() in app.py passes settings.llm_chat_completions_path."""
        from voice_intake.api.app import _build_llm_client
        from voice_intake.config import Settings

        # Test default path
        settings = Settings(
            mock_llm=False,
            llm_provider="remote",
            llm_base_url="https://api.example.com",
            llm_model="test-model",
            llm_api_key="test-key",
        )
        client = _build_llm_client(settings)
        self.assertIsNotNone(client)
        self.assertEqual(client._chat_completions_path, "/v1/chat/completions")

        # Test custom path
        settings_gemini = Settings(
            mock_llm=False,
            llm_provider="remote",
            llm_base_url="https://api.example.com",
            llm_model="gemini-2.5-flash-lite",
            llm_api_key="test-key",
            llm_chat_completions_path="/chat/completions",
        )
        client_gemini = _build_llm_client(settings_gemini)
        self.assertIsNotNone(client_gemini)
        self.assertEqual(client_gemini._chat_completions_path, "/chat/completions")


class LLMClientMaxRetriesTest(unittest.TestCase):
    """LLM_MAX_RETRIES must actually bound retry attempts (was hardcoded to 3)."""

    def test_default_max_retries_is_three(self):
        from voice_intake.llm.client import LLMClient

        client = LLMClient(base_url="https://api.example.com", model="test-model")
        self.assertEqual(client._max_retries, 3)

    def test_custom_max_retries_stored(self):
        from voice_intake.llm.client import LLMClient

        client = LLMClient(
            base_url="https://api.example.com", model="test-model", max_retries=1
        )
        self.assertEqual(client._max_retries, 1)

    def test_max_retries_bounds_attempts(self):
        """max_retries=1 means exactly 2 attempts (initial + one retry) on timeout."""
        from unittest.mock import patch
        from voice_intake.llm.client import LLMClient

        client = LLMClient(
            base_url="https://api.example.com", model="test-model", max_retries=1
        )
        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.TimeoutException("simulated timeout")
            result = _run(client.propose(
                {"session_id": "s-1", "current_state": "opening", "recent_turns": []},
                "turn-1",
            ))
        self.assertIsNone(result)
        self.assertEqual(mock_post.call_count, 2)

    def test_ollama_client_max_retries_stored(self):
        from voice_intake.llm.ollama_client import OllamaClient

        client = OllamaClient(
            base_url="http://localhost:11434", model="llama3.1:8b", max_retries=2
        )
        self.assertEqual(client._max_retries, 2)

    def test_build_llm_client_passes_max_retries_from_settings(self):
        from voice_intake.api.app import _build_llm_client
        from voice_intake.config import Settings

        settings = Settings(
            mock_llm=False,
            llm_provider="remote",
            llm_base_url="https://api.example.com",
            llm_model="test-model",
            llm_api_key="test-key",
            llm_max_retries=1,
        )
        client = _build_llm_client(settings)
        self.assertIsNotNone(client)
        self.assertEqual(client._max_retries, 1)

        settings_ollama = Settings(
            mock_llm=False,
            llm_provider="ollama",
            llm_max_retries=2,
        )
        client_ollama = _build_llm_client(settings_ollama)
        self.assertIsNotNone(client_ollama)
        self.assertEqual(client_ollama._max_retries, 2)


if __name__ == "__main__":
    unittest.main()
