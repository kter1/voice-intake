"""
Integration tests for the HTTP API layer.
Uses FastAPI TestClient with in-memory SQLite and MOCK_LLM=true.
"""
import unittest

import pytest
from fastapi.testclient import TestClient

from voice_intake.api.app import create_app
from voice_intake.api.deps import get_store
from voice_intake.config import Settings
from voice_intake.db import SQLAuditStore, create_tables, get_engine, make_session_factory


def _mock_embed(texts: list[str]) -> list[list[float]]:
    return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


def _make_test_app():
    # Override settings: in-memory SQLite + mock LLM/ASR/RAG
    test_settings = Settings(
        database_url="sqlite:///:memory:",
        mock_llm=True,
        mock_asr=True,
        mock_rag=True,
        audit_hmac_secret="test-audit-secret",
        stream_auth_secret="test-stream-secret",
    )
    app = create_app(settings=test_settings)
    app.state.embed_fn = _mock_embed
    engine = get_engine("sqlite:///:memory:")
    create_tables(engine)
    factory = make_session_factory(engine)
    store = SQLAuditStore(factory, hmac_secret="test-audit-secret")

    app.dependency_overrides[get_store] = lambda: store
    app.state.session_factory = factory

    return app, store


class SessionApiTest(unittest.TestCase):
    def setUp(self):
        app, self.store = _make_test_app()
        self.client = TestClient(app, raise_server_exceptions=True)

    def test_open_session_returns_201(self):
        resp = self.client.post("/session/open", json={
            "telephony_call_id": "call-001",
            "operator_id": "op-1",
            "stir_shaken_attestation": "A",
        })
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertIn("session_id", data)
        self.assertIn("current_state", data)

    def test_open_session_state_is_consent_recording(self):
        resp = self.client.post("/session/open", json={
            "telephony_call_id": "call-002",
            "operator_id": "op-1",
        })
        self.assertEqual(resp.status_code, 201)
        # Default policy requires recording consent first
        self.assertEqual(resp.json()["current_state"], "consent_recording")

    def test_open_session_persists_to_db(self):
        resp = self.client.post("/session/open", json={
            "telephony_call_id": "call-003",
            "operator_id": "op-1",
        })
        session_id = resp.json()["session_id"]
        loaded = self.store.load_session(session_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.operator_id, "op-1")

    def test_get_session_returns_200(self):
        resp = self.client.post("/session/open", json={
            "telephony_call_id": "call-004",
            "operator_id": "op-1",
        })
        session_id = resp.json()["session_id"]
        get_resp = self.client.get(f"/session/{session_id}")
        self.assertEqual(get_resp.status_code, 200)
        self.assertEqual(get_resp.json()["session_id"], session_id)

    def test_get_nonexistent_session_returns_404(self):
        resp = self.client.get("/session/nonexistent-id")
        self.assertEqual(resp.status_code, 404)

    def test_list_sessions(self):
        for i in range(3):
            self.client.post("/session/open", json={
                "telephony_call_id": f"call-{i}",
                "operator_id": "op-1",
            })
        resp = self.client.get("/sessions")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.json()["sessions"]), 3)

    def test_unknown_attestation_falls_back_to_unknown(self):
        resp = self.client.post("/session/open", json={
            "telephony_call_id": "call-005",
            "operator_id": "op-1",
            "stir_shaken_attestation": "BOGUS",
        })
        self.assertEqual(resp.status_code, 201)


class ConsentApiTest(unittest.TestCase):
    def setUp(self):
        app, self.store = _make_test_app()
        self.client = TestClient(app, raise_server_exceptions=True)
        resp = self.client.post("/session/open", json={
            "telephony_call_id": "call-consent",
            "operator_id": "op-1",
        })
        self.session_id = resp.json()["session_id"]

    def test_recording_consent_granted_advances_state(self):
        resp = self.client.post(f"/session/{self.session_id}/consent", json={
            "consent_type": "recording_consent",
            "granted": True,
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["next_state"], "consent_ai_assistance")

    def test_recording_consent_declined_still_advances(self):
        resp = self.client.post(f"/session/{self.session_id}/consent", json={
            "consent_type": "recording_consent",
            "granted": False,
        })
        self.assertEqual(resp.status_code, 200)
        # allow_ai_without_recording=True by default → still goes to consent_ai_assistance
        self.assertEqual(resp.json()["next_state"], "consent_ai_assistance")

    def test_invalid_consent_type_returns_422(self):
        resp = self.client.post(f"/session/{self.session_id}/consent", json={
            "consent_type": "invalid_type",
            "granted": True,
        })
        self.assertEqual(resp.status_code, 422)

    def test_consent_state_persisted_to_db(self):
        self.client.post(f"/session/{self.session_id}/consent", json={
            "consent_type": "recording_consent",
            "granted": True,
        })
        loaded = self.store.load_session(self.session_id)
        self.assertEqual(loaded.current_state.value, "consent_ai_assistance")

    def test_both_consents_granted_reaches_identity_capture(self):
        self.client.post(f"/session/{self.session_id}/consent", json={
            "consent_type": "recording_consent",
            "granted": True,
        })
        resp = self.client.post(f"/session/{self.session_id}/consent", json={
            "consent_type": "ai_assistance_consent",
            "granted": True,
        })
        self.assertEqual(resp.json()["next_state"], "identity_capture")


class TurnApiTest(unittest.TestCase):
    def setUp(self):
        app, self.store = _make_test_app()
        self.client = TestClient(app, raise_server_exceptions=True)
        # Open session and grant both consents so we're in identity_capture
        resp = self.client.post("/session/open", json={
            "telephony_call_id": "call-turn",
            "operator_id": "op-1",
        })
        self.session_id = resp.json()["session_id"]
        self.client.post(f"/session/{self.session_id}/consent", json={
            "consent_type": "recording_consent", "granted": True,
        })
        self.client.post(f"/session/{self.session_id}/consent", json={
            "consent_type": "ai_assistance_consent", "granted": True,
        })

    def test_turn_returns_200_with_turn_id(self):
        resp = self.client.post(f"/session/{self.session_id}/turn", json={
            "transcript": "My name is Jane Doe",
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("turn_id", data)
        self.assertIn("current_state", data)

    def test_turn_records_to_db(self):
        self.client.post(f"/session/{self.session_id}/turn", json={
            "transcript": "My name is Jane Doe",
        })
        turns = self.store.turns_for_session(self.session_id)
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0].transcript, "My name is Jane Doe")

    def test_turn_on_nonexistent_session_returns_404(self):
        resp = self.client.post("/session/bad-id/turn", json={"transcript": "hello"})
        self.assertEqual(resp.status_code, 404)

    def test_turn_state_persisted_after_accepted_proposal(self):
        self.client.post(f"/session/{self.session_id}/turn", json={
            "transcript": "My name is Jane",
        })
        loaded = self.store.load_session(self.session_id)
        self.assertIsNotNone(loaded)


class AuditApiTest(unittest.TestCase):
    def setUp(self):
        app, self.store = _make_test_app()
        self.client = TestClient(app, raise_server_exceptions=True)
        resp = self.client.post("/session/open", json={
            "telephony_call_id": "call-audit",
            "operator_id": "op-1",
        })
        self.session_id = resp.json()["session_id"]

    def test_audit_returns_events(self):
        resp = self.client.get(f"/session/{self.session_id}/audit")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["session_id"], self.session_id)
        self.assertGreater(len(data["events"]), 0)

    def test_audit_events_have_required_fields(self):
        resp = self.client.get(f"/session/{self.session_id}/audit")
        event = resp.json()["events"][0]
        self.assertIn("event_id", event)
        self.assertIn("event_type", event)
        self.assertIn("actor", event)
        self.assertIn("timestamp", event)

    def test_audit_does_not_expose_details(self):
        resp = self.client.get(f"/session/{self.session_id}/audit")
        event = resp.json()["events"][0]
        self.assertNotIn("details", event)

    def test_audit_on_nonexistent_session_returns_404(self):
        resp = self.client.get("/session/bad-id/audit")
        self.assertEqual(resp.status_code, 404)

    def test_audit_verify_returns_valid_for_fresh_session(self):
        resp = self.client.get(f"/session/{self.session_id}/audit/verify")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["valid"])
        self.assertEqual(data["errors"], [])
        self.assertGreater(data["events_verified"], 0)

    def test_audit_verify_returns_invalid_after_tamper(self):
        from voice_intake.db.models_orm import AuditEventORM

        with self.store._session() as db:
            row = (
                db.query(AuditEventORM)
                .filter(AuditEventORM.session_id == self.session_id)
                .first()
            )
            self.assertIsNotNone(row)
            row.chain_hash = "tampered"
            db.commit()

        resp = self.client.get(f"/session/{self.session_id}/audit/verify")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["valid"])
        self.assertGreater(len(data["errors"]), 0)


class SupervisorApiTest(unittest.TestCase):
    def setUp(self):
        app, self.store = _make_test_app()
        self.client = TestClient(app, raise_server_exceptions=True)
        resp = self.client.post("/session/open", json={
            "telephony_call_id": "call-sup",
            "operator_id": "op-1",
        })
        self.session_id = resp.json()["session_id"]

    def test_forced_takeover_transitions_state(self):
        resp = self.client.post(f"/session/{self.session_id}/supervisor/intervene", json={
            "intervention_type": "forced_takeover",
            "reason_code": "test_reason",
            "notes": "Testing forced takeover.",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["current_state"], "human_takeover")

    def test_forced_takeover_persisted_to_db(self):
        self.client.post(f"/session/{self.session_id}/supervisor/intervene", json={
            "intervention_type": "forced_takeover",
            "reason_code": "test_reason",
        })
        loaded = self.store.load_session(self.session_id)
        self.assertEqual(loaded.current_state.value, "human_takeover")

    def test_invalid_intervention_type_returns_422(self):
        resp = self.client.post(f"/session/{self.session_id}/supervisor/intervene", json={
            "intervention_type": "explode",
            "reason_code": "test",
        })
        self.assertEqual(resp.status_code, 422)

    def test_intervention_on_nonexistent_session_returns_404(self):
        resp = self.client.post("/session/bad-id/supervisor/intervene", json={
            "intervention_type": "forced_takeover",
            "reason_code": "test",
        })
        self.assertEqual(resp.status_code, 404)


# ---------------------------------------------------------------------------
# Module-level pytest tests (need monkeypatch / xfail which don't fit
# TestCase classes cleanly).
# ---------------------------------------------------------------------------


def _open_session_with_both_consents(client) -> str:
    """Open a session and grant recording + AI consents → state=identity_capture."""
    resp = client.post("/session/open", json={
        "telephony_call_id": "call-happy",
        "operator_id": "op-1",
    })
    session_id = resp.json()["session_id"]
    client.post(f"/session/{session_id}/consent", json={
        "consent_type": "recording_consent", "granted": True,
    })
    client.post(f"/session/{session_id}/consent", json={
        "consent_type": "ai_assistance_consent", "granted": True,
    })
    return session_id


def test_full_mock_happy_path_no_rejections():
    """Full intake flow through mock LLM: no validator rejections from identity_capture
    through close. Consent is granted via /consent (not driven by /turn)."""
    app, store = _make_test_app()
    client = TestClient(app, raise_server_exceptions=True)
    session_id = _open_session_with_both_consents(client)

    # Sanity-check starting state
    loaded = store.load_session(session_id)
    assert loaded.current_state.value == "identity_capture"

    # Loop /turn through field-collection states. Each MockLLM fallback advances
    # exactly one state forward. We stop before submitting a /turn from "close"
    # (CLOSE has no allowed transitions).
    expected_path = [
        "demographics",
        "insurance",
        "reason_for_visit",
        "readback",
        "disposition",
        "close",
    ]
    for i, expected_next in enumerate(expected_path):
        resp = client.post(f"/session/{session_id}/turn", json={
            "transcript": f"Mock response {i}",
        })
        assert resp.status_code == 200, f"step {i}: HTTP {resp.status_code} body={resp.json()}"
        data = resp.json()
        assert data.get("rejection") is None, f"step {i}: rejection={data['rejection']}"
        assert data["current_state"] == expected_next, (
            f"step {i}: expected {expected_next}, got {data['current_state']}"
        )


def test_timeout_escalates_to_human_takeover(monkeypatch):
    """When propose_next_action returns None, the orchestrator must set HUMAN_TAKEOVER."""
    import voice_intake.api.routers.turns as turns_module

    async def _fake_timeout(*args, **kwargs):
        return None

    monkeypatch.setattr(turns_module, "propose_next_action", _fake_timeout)

    app, store = _make_test_app()
    client = TestClient(app, raise_server_exceptions=True)
    session_id = _open_session_with_both_consents(client)

    resp = client.post(f"/session/{session_id}/turn", json={
        "transcript": "anything",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["current_state"] == "human_takeover"
    assert data.get("rejection", {}).get("reason") == "llm_timeout"

    # TurnResponse doesn't expose mode - load from store to confirm.
    loaded = store.load_session(session_id)
    assert loaded.mode.value == "human_takeover"


def test_double_reject_across_separate_turns_escalates(monkeypatch):
    """Two separate /turn calls (each with its own turn_id) returning invalid proposals
    must escalate to HUMAN_TAKEOVER. State-keyed counter, not turn-keyed."""
    import voice_intake.api.routers.turns as turns_module
    from voice_intake.models import CallState, ModelProposal
    from uuid import uuid4

    async def _fake_invalid(
        prompt,
        source_turn_id,
        settings,
        llm_client,
        demo_router=None,
        safety_pre_router=None,
        proposal_id=None,
    ):
        # collect_field_prompt with empty variables → VARIABLE_SCHEMA_VIOLATION
        return ModelProposal(
            proposal_id=str(uuid4()),
            source_turn_id=source_turn_id,
            template_id="collect_field_prompt",
            variables={},
            requested_transition=CallState.DEMOGRAPHICS,
            model_version="fake-invalid",
        )

    monkeypatch.setattr(turns_module, "propose_next_action", _fake_invalid)

    app, store = _make_test_app()
    client = TestClient(app, raise_server_exceptions=True)
    session_id = _open_session_with_both_consents(client)

    resp1 = client.post(f"/session/{session_id}/turn", json={"transcript": "first"})
    assert resp1.status_code == 200
    assert resp1.json()["current_state"] == "identity_capture"  # not yet escalated

    resp2 = client.post(f"/session/{session_id}/turn", json={"transcript": "second"})
    assert resp2.status_code == 200
    assert resp2.json()["current_state"] == "human_takeover"


def test_mock_rag_dependency_returns_none():
    """When settings.mock_rag is true, get_knowledge_retriever must return None and
    never reconstruct KnowledgeRetriever from chroma_client. Without this guard, the
    lifespan-level mock would be silently bypassed."""
    from types import SimpleNamespace
    from voice_intake.api.deps import get_knowledge_retriever

    app, _store = _make_test_app()
    app.state.settings.mock_rag = True
    app.state.retriever = None
    request = SimpleNamespace(app=app)
    assert get_knowledge_retriever(request) is None


@pytest.mark.xfail(
    reason="/turn currently advances consent states without recording a ConsentArtifact",
    strict=True,
)
def test_turn_does_not_leave_consent_state_without_consent_artifact():
    """Desired invariant: a session in a consent state cannot advance through /turn
    without producing a ConsentArtifact.

    This currently fails because /turn drives state via the LLM proposal regardless of
    consent state. When the architectural fix lands (block /turn during consent OR route
    consent transcripts through capture_consent), this test will pass and strict=True
    will then fail the run, prompting removal of the marker."""
    app, store = _make_test_app()
    client = TestClient(app, raise_server_exceptions=True)

    resp = client.post("/session/open", json={
        "telephony_call_id": "call-consent-turn",
        "operator_id": "op-1",
    })
    session_id = resp.json()["session_id"]

    client.post(f"/session/{session_id}/turn", json={
        "transcript": "Yes, you can record the call",
    })

    loaded = store.load_session(session_id)
    consents = store.consents_for_session(session_id)

    # Desired invariant: either state stays in consent_recording, or an artifact exists.
    assert (
        loaded.current_state.value == "consent_recording"
        or len(consents) > 0
    )


class TemplateFilteringTest(unittest.TestCase):
    """Test that available_templates are filtered by current_state.allowed_states."""

    def setUp(self):
        app, self.store = _make_test_app()
        self.app = app
        self.client = TestClient(app, raise_server_exceptions=True)

    def _get_templates_for_state(self, state_value: str):
        """Simulate the filtering logic from turns.py to see what templates
        would be available for a given state."""
        from voice_intake.models import CallState
        from voice_intake.templates import DEFAULT_TEMPLATE_BUNDLE

        current_state = CallState(state_value)

        available_templates = [
            template
            for template in DEFAULT_TEMPLATE_BUNDLE.templates.values()
            if current_state in template.allowed_states
        ]
        return available_templates

    def test_opening_state_exposes_opening_disclosure(self):
        """From OPENING state, opening_disclosure should be available."""
        templates = self._get_templates_for_state("opening")
        template_ids = [t.template_id for t in templates]
        self.assertIn("opening_disclosure", template_ids)

    def test_opening_state_does_not_expose_ai_assistance_consent_prompt(self):
        """From OPENING state, ai_assistance_consent_prompt should NOT be available
        because it's only allowed in CONSENT_AI_ASSISTANCE state."""
        templates = self._get_templates_for_state("opening")
        template_ids = [t.template_id for t in templates]
        self.assertNotIn("ai_assistance_consent_prompt", template_ids)

    def test_opening_state_does_not_expose_consent_prompts(self):
        """From OPENING state, recording/AI consent prompts should NOT be available."""
        templates = self._get_templates_for_state("opening")
        template_ids = [t.template_id for t in templates]
        self.assertNotIn("recording_consent_prompt", template_ids)
        self.assertNotIn("ai_assistance_consent_prompt", template_ids)

    def test_consent_recording_state_exposes_recording_consent_prompt(self):
        """From CONSENT_RECORDING state, recording_consent_prompt should be available."""
        templates = self._get_templates_for_state("consent_recording")
        template_ids = [t.template_id for t in templates]
        self.assertIn("recording_consent_prompt", template_ids)

    def test_consent_recording_state_does_not_expose_opening_disclosure(self):
        """From CONSENT_RECORDING state, opening_disclosure should NOT be available."""
        templates = self._get_templates_for_state("consent_recording")
        template_ids = [t.template_id for t in templates]
        self.assertNotIn("opening_disclosure", template_ids)

    def test_identity_capture_state_exposes_collect_field_prompt(self):
        """From IDENTITY_CAPTURE state, collect_field_prompt should be available
        because it allows IDENTITY_CAPTURE."""
        templates = self._get_templates_for_state("identity_capture")
        template_ids = [t.template_id for t in templates]
        self.assertIn("collect_field_prompt", template_ids)

    def test_identity_capture_does_not_expose_ai_assistance_consent_prompt(self):
        """From IDENTITY_CAPTURE state, ai_assistance_consent_prompt should NOT be available."""
        templates = self._get_templates_for_state("identity_capture")
        template_ids = [t.template_id for t in templates]
        self.assertNotIn("ai_assistance_consent_prompt", template_ids)

    def test_only_matching_states_exposed(self):
        """Verify that ALL exposed templates have current_state in their allowed_states."""
        from voice_intake.models import CallState

        for state in [CallState.OPENING, CallState.CONSENT_RECORDING, CallState.IDENTITY_CAPTURE]:
            templates = self._get_templates_for_state(state.value)
            for template in templates:
                self.assertIn(state, template.allowed_states,
                    f"Template {template.template_id} exposed for {state.value} "
                    f"but its allowed_states are {template.allowed_states}")


def _make_test_app_with_settings(test_settings):
    """Variant of _make_test_app that takes a fully-customized Settings object."""
    app = create_app(settings=test_settings)
    app.state.embed_fn = _mock_embed
    engine = get_engine("sqlite:///:memory:")
    create_tables(engine)
    factory = make_session_factory(engine)
    store = SQLAuditStore(factory, hmac_secret="test-audit-secret")
    app.dependency_overrides[get_store] = lambda: store
    app.state.session_factory = factory
    return app, store


class PolicyProfileSettingsTest(unittest.TestCase):
    """Settings.policy_profile decoupling from demo_router (C4.3)."""

    def test_default_policy_profile_is_default(self):
        s = Settings(audit_hmac_secret="x", stream_auth_secret="y")
        self.assertEqual(s.policy_profile, "default")

    def test_demo_policy_profile_round_trips(self):
        s = Settings(
            audit_hmac_secret="x",
            stream_auth_secret="y",
            policy_profile="demo",
        )
        self.assertEqual(s.policy_profile, "demo")


class PolicyProfileOrchestratorTest(unittest.TestCase):
    """get_orchestrator() uses combined demo-policy predicate (C4.3)."""

    def _orchestrator_profile_id(self, *, policy_profile, demo_router):
        from voice_intake.api.deps import get_orchestrator

        settings = Settings(
            database_url="sqlite:///:memory:",
            mock_llm=True,
            mock_asr=True,
            mock_rag=True,
            audit_hmac_secret="test-audit-secret",
            stream_auth_secret="test-stream-secret",
            policy_profile=policy_profile,
            demo_router=demo_router,
        )
        app, store = _make_test_app_with_settings(settings)
        # Build a minimal Request-like object: get_orchestrator only needs settings + store.
        class _Req:
            pass
        req = _Req()
        req.app = app
        orchestrator = get_orchestrator(req, store=store, settings=settings)
        return orchestrator.policy_profile.policy_profile_id

    def test_demo_profile_with_demo_router_off_uses_demo_policy(self):
        self.assertEqual(
            self._orchestrator_profile_id(policy_profile="demo", demo_router=False),
            "demo",
        )

    def test_default_profile_with_demo_router_on_uses_demo_policy(self):
        # Backward-compat shim: DEMO_ROUTER=true still implies demo policy.
        self.assertEqual(
            self._orchestrator_profile_id(policy_profile="default", demo_router=True),
            "demo",
        )

    def test_default_profile_with_demo_router_off_uses_default_policy(self):
        self.assertEqual(
            self._orchestrator_profile_id(policy_profile="default", demo_router=False),
            "default",
        )

    def test_demo_profile_with_demo_router_on_uses_demo_policy(self):
        self.assertEqual(
            self._orchestrator_profile_id(policy_profile="demo", demo_router=True),
            "demo",
        )


class PolicyProfileApiTest(unittest.TestCase):
    """End-to-end: POLICY_PROFILE=demo opens session at OPENING (C4.3)."""

    def test_demo_profile_opens_session_at_opening(self):
        settings = Settings(
            database_url="sqlite:///:memory:",
            mock_llm=True,
            mock_asr=True,
            mock_rag=True,
            audit_hmac_secret="test-audit-secret",
            stream_auth_secret="test-stream-secret",
            policy_profile="demo",
        )
        app, _ = _make_test_app_with_settings(settings)
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post("/session/open", json={
            "telephony_call_id": "call-policy-demo",
            "operator_id": "op-1",
        })
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(resp.json()["current_state"], "opening")

    def test_default_profile_still_opens_at_consent_recording(self):
        # Regression guard: default behavior unchanged.
        settings = Settings(
            database_url="sqlite:///:memory:",
            mock_llm=True,
            mock_asr=True,
            mock_rag=True,
            audit_hmac_secret="test-audit-secret",
            stream_auth_secret="test-stream-secret",
            policy_profile="default",
            demo_router=False,
        )
        app, _ = _make_test_app_with_settings(settings)
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post("/session/open", json={
            "telephony_call_id": "call-policy-default",
            "operator_id": "op-1",
        })
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(resp.json()["current_state"], "consent_recording")


def _demo_profile_settings() -> Settings:
    return Settings(
        database_url="sqlite:///:memory:",
        mock_llm=True,
        mock_asr=True,
        mock_rag=True,
        audit_hmac_secret="test-audit-secret",
        stream_auth_secret="test-stream-secret",
        policy_profile="demo",
    )


def test_demo_profile_first_timeout_holds_then_second_escalates(monkeypatch):
    """Demo profile: first LLM timeout emits hold_message and keeps the session
    alive; a second consecutive timeout escalates to HUMAN_TAKEOVER."""
    import voice_intake.api.routers.turns as turns_module

    async def _fake_timeout(*args, **kwargs):
        return None

    monkeypatch.setattr(turns_module, "propose_next_action", _fake_timeout)

    app, store = _make_test_app_with_settings(_demo_profile_settings())
    client = TestClient(app, raise_server_exceptions=True)
    resp = client.post("/session/open", json={
        "telephony_call_id": "call-grace", "operator_id": "op-1",
    })
    session_id = resp.json()["session_id"]
    assert resp.json()["current_state"] == "opening"

    resp1 = client.post(f"/session/{session_id}/turn", json={
        "transcript": "purple elephant unmatched input",
    })
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["current_state"] == "opening"  # not escalated
    assert data1.get("rejection") is None
    assert data1["voice_action"]["template_id"] == "hold_message"

    resp2 = client.post(f"/session/{session_id}/turn", json={
        "transcript": "purple elephant again",
    })
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["current_state"] == "human_takeover"
    assert data2["rejection"]["reason"] == "llm_timeout"


def test_demo_profile_timeout_counter_resets_after_success(monkeypatch):
    """A successful proposal between timeouts resets the consecutive-timeout
    counter, so the next timeout gets the grace path again."""
    import voice_intake.api.routers.turns as turns_module
    from voice_intake.models import CallState, ModelProposal
    from uuid import uuid4

    behaviors = ["timeout", "success", "timeout"]

    async def _fake(
        prompt,
        source_turn_id,
        settings,
        llm_client,
        demo_router=None,
        safety_pre_router=None,
        proposal_id=None,
    ):
        behavior = behaviors.pop(0)
        if behavior == "timeout":
            return None
        return ModelProposal(
            proposal_id=proposal_id or str(uuid4()),
            source_turn_id=source_turn_id,
            template_id="appointment_reason_prompt",
            variables={},
            requested_transition=CallState.REASON_FOR_VISIT,
            model_version="fake",
        )

    monkeypatch.setattr(turns_module, "propose_next_action", _fake)

    app, store = _make_test_app_with_settings(_demo_profile_settings())
    client = TestClient(app, raise_server_exceptions=True)
    resp = client.post("/session/open", json={
        "telephony_call_id": "call-reset", "operator_id": "op-1",
    })
    session_id = resp.json()["session_id"]

    resp1 = client.post(f"/session/{session_id}/turn", json={"transcript": "one"})
    assert resp1.json()["voice_action"]["template_id"] == "hold_message"
    assert resp1.json()["current_state"] == "opening"

    resp2 = client.post(f"/session/{session_id}/turn", json={"transcript": "two"})
    assert resp2.json()["current_state"] == "reason_for_visit"
    assert resp2.json().get("rejection") is None

    # Counter reset by the success above: this timeout gets grace, not takeover.
    resp3 = client.post(f"/session/{session_id}/turn", json={"transcript": "three"})
    assert resp3.json()["voice_action"]["template_id"] == "hold_message"
    assert resp3.json()["current_state"] == "reason_for_visit"


def test_llm_prompt_includes_conversation_history(monkeypatch):
    """The prompt passed to the LLM must include prior caller turns, with the
    current utterance last."""
    import voice_intake.api.routers.turns as turns_module
    from voice_intake.models import CallState, ModelProposal
    from uuid import uuid4

    captured = []
    transitions = [CallState.DEMOGRAPHICS, CallState.INSURANCE]

    async def _fake(
        prompt,
        source_turn_id,
        settings,
        llm_client,
        demo_router=None,
        safety_pre_router=None,
        proposal_id=None,
    ):
        captured.append(prompt)
        return ModelProposal(
            proposal_id=proposal_id or str(uuid4()),
            source_turn_id=source_turn_id,
            template_id="collect_field_prompt",
            variables={"field_label": "date of birth"},
            requested_transition=transitions.pop(0),
            model_version="fake",
        )

    monkeypatch.setattr(turns_module, "propose_next_action", _fake)

    app, store = _make_test_app()
    client = TestClient(app, raise_server_exceptions=True)
    session_id = _open_session_with_both_consents(client)

    client.post(f"/session/{session_id}/turn", json={
        "transcript": "purple elephant first utterance",
    })
    client.post(f"/session/{session_id}/turn", json={
        "transcript": "purple elephant second utterance",
    })

    assert len(captured) == 2
    transcripts = [t["transcript"] for t in captured[1]["recent_turns"]]
    assert len(transcripts) >= 2
    assert any("first utterance" in t for t in transcripts)
    assert "second utterance" in transcripts[-1]


def _spoken_text_proposal_factory(spoken_text):
    from voice_intake.models import CallState, ModelProposal
    from uuid import uuid4

    async def _fake(
        prompt,
        source_turn_id,
        settings,
        llm_client,
        demo_router=None,
        safety_pre_router=None,
        proposal_id=None,
    ):
        return ModelProposal(
            proposal_id=proposal_id or str(uuid4()),
            source_turn_id=source_turn_id,
            template_id="appointment_reason_prompt",
            variables={},
            requested_transition=CallState.REASON_FOR_VISIT,
            model_version="fake",
            spoken_text=spoken_text,
        )

    return _fake


def test_demo_profile_spoken_text_passes_guard_and_is_returned(monkeypatch):
    """Demo profile: guard-approved spoken_text reaches the API response."""
    import voice_intake.api.routers.turns as turns_module

    monkeypatch.setattr(
        turns_module,
        "propose_next_action",
        _spoken_text_proposal_factory(
            "Happy to help with scheduling. What is the reason for the visit?"
        ),
    )
    app, store = _make_test_app_with_settings(_demo_profile_settings())
    client = TestClient(app, raise_server_exceptions=True)
    resp = client.post("/session/open", json={
        "telephony_call_id": "call-spoken", "operator_id": "op-1",
    })
    session_id = resp.json()["session_id"]

    data = client.post(f"/session/{session_id}/turn", json={
        "transcript": "hi, my knee has been bothering me",
    }).json()
    assert data["current_state"] == "reason_for_visit"
    assert data["voice_action"]["template_id"] == "appointment_reason_prompt"
    assert data["voice_action"]["spoken_text"] == (
        "Happy to help with scheduling. What is the reason for the visit?"
    )


def test_demo_profile_guarded_spoken_text_falls_back_to_template(monkeypatch):
    """Guard rejection (clinical language) drops spoken_text but keeps the
    accepted template action."""
    import voice_intake.api.routers.turns as turns_module

    monkeypatch.setattr(
        turns_module,
        "propose_next_action",
        _spoken_text_proposal_factory("That sounds like a diagnosis of arthritis."),
    )
    app, store = _make_test_app_with_settings(_demo_profile_settings())
    client = TestClient(app, raise_server_exceptions=True)
    resp = client.post("/session/open", json={
        "telephony_call_id": "call-guarded", "operator_id": "op-1",
    })
    session_id = resp.json()["session_id"]

    data = client.post(f"/session/{session_id}/turn", json={
        "transcript": "my knee hurts",
    }).json()
    assert data["current_state"] == "reason_for_visit"
    assert data["voice_action"]["template_id"] == "appointment_reason_prompt"
    assert data["voice_action"]["spoken_text"] is None


def test_default_profile_never_emits_spoken_text(monkeypatch):
    """Default profile: spoken_text is gated off entirely, even when the model
    provides an innocuous one."""
    import voice_intake.api.routers.turns as turns_module
    from voice_intake.models import CallState, ModelProposal
    from uuid import uuid4

    async def _fake(
        prompt,
        source_turn_id,
        settings,
        llm_client,
        demo_router=None,
        safety_pre_router=None,
        proposal_id=None,
    ):
        return ModelProposal(
            proposal_id=proposal_id or str(uuid4()),
            source_turn_id=source_turn_id,
            template_id="collect_field_prompt",
            variables={"field_label": "date of birth"},
            requested_transition=CallState.DEMOGRAPHICS,
            model_version="fake",
            spoken_text="Sure, could you share your date of birth?",
        )

    monkeypatch.setattr(turns_module, "propose_next_action", _fake)
    app, store = _make_test_app()
    client = TestClient(app, raise_server_exceptions=True)
    session_id = _open_session_with_both_consents(client)

    data = client.post(f"/session/{session_id}/turn", json={
        "transcript": "my name is on file already",
    }).json()
    assert data["voice_action"]["template_id"] == "collect_field_prompt"
    assert data["voice_action"]["spoken_text"] is None


if __name__ == "__main__":
    unittest.main()
