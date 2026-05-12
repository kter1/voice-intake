import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock

import httpx

from voice_intake.models import CallState


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class OllamaClientTest(unittest.TestCase):
    def test_propose_returns_model_proposal_on_success(self):
        from voice_intake.llm.ollama_client import OllamaClient

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
                                            "template_id": "collect_field_prompt",
                                            "variables": {"field_label": "date of birth"},
                                            "requested_transition": "identity_capture",
                                        }
                                    )
                                }
                            }
                        ]
                    }
                }
            ]
        }

        client = OllamaClient("http://localhost:11434", "llama3.1:8b", timeout=1.0)
        client._client.post = AsyncMock(return_value=response)

        proposal = _run(
            client.propose({"current_state": "identity_capture", "recent_turns": []}, "turn-1", "prop-1")
        )

        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.proposal_id, "prop-1")
        self.assertEqual(proposal.template_id, "collect_field_prompt")
        self.assertEqual(proposal.variables, {"field_label": "date of birth"})
        self.assertEqual(proposal.requested_transition, CallState.IDENTITY_CAPTURE)
        _run(client.aclose())

    def test_propose_uses_recruiter_prompt_profile(self):
        from voice_intake.llm.ollama_client import OllamaClient

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

        client = OllamaClient(
            "http://localhost:11434",
            "llama3.1:8b",
            timeout=1.0,
            prompt_profile="recruiter_demo",
        )
        client._client.post = AsyncMock(return_value=response)

        proposal = _run(client.propose({"current_state": "opening"}, "turn-1"))

        self.assertIsNotNone(proposal)
        payload = client._client.post.call_args.kwargs["json"]
        system_message = payload["messages"][0]["content"]
        self.assertIn("Recruiter demo profile", system_message)
        self.assertIn("appointment scheduling intake only", system_message)
        _run(client.aclose())

    def test_propose_returns_none_on_connect_error(self):
        from voice_intake.llm.ollama_client import OllamaClient

        client = OllamaClient("http://localhost:11434", "llama3.1:8b", timeout=1.0)
        client._client.post = AsyncMock(
            side_effect=httpx.ConnectError("down", request=httpx.Request("POST", "http://localhost"))
        )

        proposal = _run(client.propose({"current_state": "identity_capture", "recent_turns": []}, "turn-1"))

        self.assertIsNone(proposal)
        _run(client.aclose())

    def test_propose_returns_none_on_missing_tool_calls(self):
        from voice_intake.llm.ollama_client import OllamaClient

        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {"choices": [{"message": {}}]}

        client = OllamaClient("http://localhost:11434", "llama3.1:8b", timeout=1.0)
        client._client.post = AsyncMock(return_value=response)

        proposal = _run(client.propose({"current_state": "identity_capture", "recent_turns": []}, "turn-1"))

        self.assertIsNone(proposal)
        _run(client.aclose())

    def test_propose_returns_none_on_empty_requested_transition(self):
        from voice_intake.llm.ollama_client import OllamaClient

        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"tool_calls": [{"function": {"arguments": json.dumps({
                "template_id": "collect_field_prompt",
                "variables": {},
                "requested_transition": "",
            })}}]}}]
        }

        client = OllamaClient("http://localhost:11434", "llama3.1:8b", timeout=1.0)
        client._client.post = AsyncMock(return_value=response)
        result = _run(client.propose({}, "turn-1"))
        _run(client.aclose())

        self.assertIsNone(result)

    def test_propose_returns_none_on_missing_requested_transition(self):
        from voice_intake.llm.ollama_client import OllamaClient

        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"tool_calls": [{"function": {"arguments": json.dumps({
                "template_id": "collect_field_prompt",
                "variables": {},
                # requested_transition deliberately omitted
            })}}]}}]
        }

        client = OllamaClient("http://localhost:11434", "llama3.1:8b", timeout=1.0)
        client._client.post = AsyncMock(return_value=response)
        result = _run(client.propose({}, "turn-1"))
        _run(client.aclose())

        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # JSON content fallback (tool_calls absent)
    # ------------------------------------------------------------------

    def _make_content_response(self, content: str) -> MagicMock:
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"content": content}}]
        }
        return response

    def test_content_fallback_plain_json_produces_proposal(self):
        from voice_intake.llm.ollama_client import OllamaClient

        content = json.dumps({
            "template_id": "collect_field_prompt",
            "variables": {"field_label": "name"},
            "requested_transition": "identity_capture",
        })
        client = OllamaClient("http://localhost:11434", "llama3.1:8b", timeout=1.0)
        client._client.post = AsyncMock(return_value=self._make_content_response(content))
        result = _run(client.propose({}, "turn-1"))
        _run(client.aclose())

        self.assertIsNotNone(result)
        self.assertEqual(result.template_id, "collect_field_prompt")
        self.assertEqual(result.requested_transition, CallState.IDENTITY_CAPTURE)

    def test_content_fallback_markdown_fenced_json_produces_proposal(self):
        from voice_intake.llm.ollama_client import OllamaClient

        args = {
            "template_id": "appointment_reason_prompt",
            "variables": {},
            "requested_transition": "reason_for_visit",
        }
        content = f"```json\n{json.dumps(args)}\n```"
        client = OllamaClient("http://localhost:11434", "llama3.1:8b", timeout=1.0)
        client._client.post = AsyncMock(return_value=self._make_content_response(content))
        result = _run(client.propose({}, "turn-1"))
        _run(client.aclose())

        self.assertIsNotNone(result)
        self.assertEqual(result.template_id, "appointment_reason_prompt")

    def test_content_fallback_prose_returns_none(self):
        from voice_intake.llm.ollama_client import OllamaClient

        client = OllamaClient("http://localhost:11434", "llama3.1:8b", timeout=1.0)
        client._client.post = AsyncMock(return_value=self._make_content_response(
            "I am an AI intake assistant. How can I help you today?"
        ))
        result = _run(client.propose({}, "turn-1"))
        _run(client.aclose())

        self.assertIsNone(result)

    def test_content_fallback_malformed_json_returns_none(self):
        from voice_intake.llm.ollama_client import OllamaClient

        client = OllamaClient("http://localhost:11434", "llama3.1:8b", timeout=1.0)
        client._client.post = AsyncMock(return_value=self._make_content_response(
            '{"template_id": "foo", "variables": {bad json}'
        ))
        result = _run(client.propose({}, "turn-1"))
        _run(client.aclose())

        self.assertIsNone(result)

    def test_content_fallback_json_missing_requested_transition_returns_none(self):
        from voice_intake.llm.ollama_client import OllamaClient

        content = json.dumps({"template_id": "collect_field_prompt", "variables": {}})
        client = OllamaClient("http://localhost:11434", "llama3.1:8b", timeout=1.0)
        client._client.post = AsyncMock(return_value=self._make_content_response(content))
        result = _run(client.propose({}, "turn-1"))
        _run(client.aclose())

        self.assertIsNone(result)

    def test_content_fallback_json_empty_requested_transition_returns_none(self):
        from voice_intake.llm.ollama_client import OllamaClient

        content = json.dumps({
            "template_id": "collect_field_prompt",
            "variables": {},
            "requested_transition": "",
        })
        client = OllamaClient("http://localhost:11434", "llama3.1:8b", timeout=1.0)
        client._client.post = AsyncMock(return_value=self._make_content_response(content))
        result = _run(client.propose({}, "turn-1"))
        _run(client.aclose())

        self.assertIsNone(result)

    def test_content_fallback_no_content_returns_none(self):
        from voice_intake.llm.ollama_client import OllamaClient

        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {"choices": [{"message": {}}]}

        client = OllamaClient("http://localhost:11434", "llama3.1:8b", timeout=1.0)
        client._client.post = AsyncMock(return_value=response)
        result = _run(client.propose({}, "turn-1"))
        _run(client.aclose())

        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # Dynamic template constraint in Ollama system prompt
    # ------------------------------------------------------------------

    def _make_success_response(self, template_id: str = "opening_disclosure",
                               transition: str = "opening") -> MagicMock:
        return self._make_content_response(json.dumps({
            "template_id": template_id,
            "variables": {},
            "requested_transition": transition,
        }))

    def test_system_prompt_lists_available_templates_exactly(self):
        from voice_intake.llm.ollama_client import OllamaClient

        templates = ["opening_disclosure", "appointment_reason_prompt", "patient_name_prompt"]
        client = OllamaClient("http://localhost:11434", "llama3.1:8b", timeout=1.0)
        client._client.post = AsyncMock(return_value=self._make_success_response())
        _run(client.propose({"available_templates": templates}, "turn-1"))
        _run(client.aclose())

        system_msg = client._client.post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertIn("valid template_id values are EXACTLY", system_msg)
        for t in templates:
            self.assertIn(t, system_msg)

    def test_system_prompt_includes_opening_disclosure_and_appointment_prompt(self):
        from voice_intake.llm.ollama_client import OllamaClient

        templates = ["opening_disclosure", "appointment_reason_prompt"]
        client = OllamaClient("http://localhost:11434", "llama3.1:8b", timeout=1.0)
        client._client.post = AsyncMock(return_value=self._make_success_response())
        _run(client.propose({"available_templates": templates}, "turn-1"))
        _run(client.aclose())

        system_msg = client._client.post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertIn("opening_disclosure", system_msg)
        self.assertIn("appointment_reason_prompt", system_msg)

    def test_system_prompt_warns_do_not_invent_template_id(self):
        from voice_intake.llm.ollama_client import OllamaClient

        client = OllamaClient("http://localhost:11434", "llama3.1:8b", timeout=1.0)
        client._client.post = AsyncMock(return_value=self._make_success_response())
        _run(client.propose({"available_templates": ["opening_disclosure"]}, "turn-1"))
        _run(client.aclose())

        system_msg = client._client.post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertIn("Do not invent template_id values", system_msg)

    def test_system_prompt_explicitly_warns_against_identity_prompt(self):
        from voice_intake.llm.ollama_client import OllamaClient

        client = OllamaClient("http://localhost:11434", "llama3.1:8b", timeout=1.0)
        client._client.post = AsyncMock(return_value=self._make_success_response())
        _run(client.propose({"available_templates": ["opening_disclosure"]}, "turn-1"))
        _run(client.aclose())

        system_msg = client._client.post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertIn("identity_prompt", system_msg)
        self.assertIn("Do not output identity_prompt", system_msg)

    def test_content_fallback_invented_template_id_reaches_validator_not_crashed(self):
        """identity_prompt is not auto-corrected; proposal is returned with invented id."""
        from voice_intake.llm.ollama_client import OllamaClient

        content = json.dumps({
            "template_id": "identity_prompt",
            "variables": {},
            "requested_transition": "identity_capture",
        })
        client = OllamaClient("http://localhost:11434", "llama3.1:8b", timeout=1.0)
        client._client.post = AsyncMock(return_value=self._make_content_response(content))
        result = _run(client.propose({}, "turn-1"))
        _run(client.aclose())

        # The client should pass it through (not crash); validator downstream rejects it.
        # An invented template_id is NOT a ValueError - coerce_transition succeeds if
        # requested_transition is a valid CallState. Validator is the correct rejection layer.
        self.assertIsNotNone(result)
        self.assertEqual(result.template_id, "identity_prompt")

    def test_no_template_constraint_when_available_templates_absent(self):
        from voice_intake.llm.ollama_client import OllamaClient

        client = OllamaClient("http://localhost:11434", "llama3.1:8b", timeout=1.0)
        client._client.post = AsyncMock(return_value=self._make_success_response())
        _run(client.propose({}, "turn-1"))  # no available_templates key
        _run(client.aclose())

        system_msg = client._client.post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertNotIn("valid template_id values are EXACTLY", system_msg)

    def test_system_prompt_includes_valid_callstate_list(self):
        """Verify Ollama system prompt lists all valid CallState values."""
        from voice_intake.llm.ollama_client import OllamaClient
        from voice_intake.models import CallState

        client = OllamaClient("http://localhost:11434", "llama3.1:8b", timeout=1.0)
        client._client.post = AsyncMock(
            return_value=self._make_content_response('{"template_id":"foo","variables":{},"requested_transition":"opening"}')
        )
        _run(client.propose({"current_state": "opening", "available_templates": ["foo"]}, "turn-1"))
        _run(client.aclose())

        # Verify the system prompt sent to Ollama contains all CallState values
        payload = client._client.post.call_args.kwargs["json"]
        system_msg = payload["messages"][0]["content"]
        for state in CallState:
            self.assertIn(state.value, system_msg, f"CallState {state.value} not in Ollama system prompt")

    def test_system_prompt_warns_against_template_id_in_transition(self):
        """Verify Ollama system prompt warns about putting template IDs in requested_transition."""
        from voice_intake.llm.ollama_client import OllamaClient

        client = OllamaClient("http://localhost:11434", "llama3.1:8b", timeout=1.0)
        client._client.post = AsyncMock(
            return_value=self._make_content_response('{"template_id":"foo","variables":{},"requested_transition":"opening"}')
        )
        _run(client.propose({}, "turn-1"))
        _run(client.aclose())

        payload = client._client.post.call_args.kwargs["json"]
        system_msg = payload["messages"][0]["content"]
        self.assertIn("template_id", system_msg)
        self.assertIn("requested_transition", system_msg)
        self.assertIn("never put a template_id", system_msg.lower())

    def test_system_prompt_includes_json_example(self):
        """Verify Ollama system prompt includes the concrete JSON example."""
        from voice_intake.llm.ollama_client import OllamaClient

        client = OllamaClient("http://localhost:11434", "llama3.1:8b", timeout=1.0)
        client._client.post = AsyncMock(
            return_value=self._make_content_response('{"template_id":"foo","variables":{},"requested_transition":"opening"}')
        )
        _run(client.propose({}, "turn-1"))
        _run(client.aclose())

        payload = client._client.post.call_args.kwargs["json"]
        system_msg = payload["messages"][0]["content"]
        self.assertIn("appointment_reason_prompt", system_msg)
        self.assertIn("reason_for_visit", system_msg)

    def test_content_fallback_template_id_in_transition_returns_none(self):
        """When model puts template_id in requested_transition, coerce_transition raises ValueError."""
        from voice_intake.llm.ollama_client import OllamaClient

        content = json.dumps({
            "template_id": "collect_field_prompt",
            "variables": {},
            "requested_transition": "collect_field_prompt",  # WRONG: is a template ID, not a CallState
        })
        client = OllamaClient("http://localhost:11434", "llama3.1:8b", timeout=1.0)
        client._client.post = AsyncMock(return_value=self._make_content_response(content))
        result = _run(client.propose({}, "turn-1"))
        _run(client.aclose())

        self.assertIsNone(result)

    def test_propose_propagates_non_transport_errors(self):
        from voice_intake.llm.ollama_client import OllamaClient

        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.side_effect = TypeError("bad response object")

        client = OllamaClient("http://localhost:11434", "llama3.1:8b", timeout=1.0)
        client._client.post = AsyncMock(return_value=response)

        with self.assertRaises(TypeError):
            _run(client.propose({"current_state": "identity_capture", "recent_turns": []}, "turn-1"))

        _run(client.aclose())

    # ─────────────────────────────────────────────────────────────────────────
    # C4.4 - allowed_transitions per-turn constraint
    # ─────────────────────────────────────────────────────────────────────────

    def test_system_prompt_includes_allowed_transitions(self):
        """Per-turn Ollama constraint must list allowed_transitions exactly."""
        from voice_intake.llm.ollama_client import OllamaClient

        client = OllamaClient("http://localhost:11434", "llama3.1:8b", timeout=1.0)
        client._client.post = AsyncMock(return_value=self._make_success_response())
        _run(client.propose(
            {
                "current_state": "opening",
                "available_templates": ["opening_disclosure"],
                "allowed_transitions": ["opening", "reason_for_visit", "human_takeover", "emergency_exit"],
            },
            "turn-1",
        ))
        _run(client.aclose())

        system_msg = client._client.post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertIn("valid requested_transition values are EXACTLY", system_msg)
        for transition in ("opening", "reason_for_visit", "human_takeover", "emergency_exit"):
            self.assertIn(transition, system_msg)

    def test_no_transition_constraint_when_allowed_transitions_absent(self):
        """When the prompt has no allowed_transitions, no transition constraint block is added."""
        from voice_intake.llm.ollama_client import OllamaClient

        client = OllamaClient("http://localhost:11434", "llama3.1:8b", timeout=1.0)
        client._client.post = AsyncMock(return_value=self._make_success_response())
        _run(client.propose(
            {"current_state": "opening", "available_templates": ["opening_disclosure"]},
            "turn-1",
        ))
        _run(client.aclose())

        system_msg = client._client.post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertNotIn("valid requested_transition values are EXACTLY", system_msg)

    def test_fallback_instruction_warns_against_consent_transitions(self):
        """Ollama fallback instruction must warn against consent/manual_mode unless listed."""
        from voice_intake.llm.ollama_client import _OLLAMA_JSON_FALLBACK_INSTRUCTION
        self.assertIn("allowed_transitions", _OLLAMA_JSON_FALLBACK_INSTRUCTION)
        self.assertIn("consent_recording", _OLLAMA_JSON_FALLBACK_INSTRUCTION)
        self.assertIn("manual_mode", _OLLAMA_JSON_FALLBACK_INSTRUCTION)
