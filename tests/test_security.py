from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from voice_intake.api.app import create_app
from voice_intake.api.security import DEV_AUDIT_SECRET, DEV_STREAM_SECRET, resolve_runtime_secrets
from voice_intake.config import Settings
from voice_intake.db import create_tables, get_engine, make_session_factory


def _mock_embed(texts: list[str]) -> list[list[float]]:
    return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


def _make_secured_app() -> tuple[object, TestClient]:
    settings = Settings(
        database_url="sqlite:///:memory:",
        mock_llm=True,
        api_key="test123",
        audit_hmac_secret="audit-secret",
        stream_auth_secret="stream-secret",
    )
    app = create_app(settings=settings)
    app.state.embed_fn = _mock_embed
    engine = get_engine("sqlite:///:memory:")
    create_tables(engine)
    app.state.session_factory = make_session_factory(engine)
    client = TestClient(app, raise_server_exceptions=False)
    return app, client


class RuntimeSecretsTest(unittest.TestCase):
    def test_create_app_populates_state_immediately(self) -> None:
        settings = Settings(
            database_url="sqlite:///:memory:",
            mock_llm=True,
            audit_hmac_secret="audit-secret",
            stream_auth_secret="stream-secret",
        )
        app = create_app(settings=settings)
        self.assertIs(app.state.settings, settings)
        self.assertEqual(app.state.audit_secret, "audit-secret")
        self.assertEqual(app.state.stream_secret, "stream-secret")
        self.assertIsNotNone(app.state.supervisor_ws_manager)

    def test_customer_pilot_requires_audit_secret(self) -> None:
        settings = Settings(
            api_key="test123",
            audit_hmac_secret="",
            stream_auth_secret="stream-secret",
        )
        with self.assertRaises(RuntimeError):
            resolve_runtime_secrets(settings)

    def test_local_mode_uses_dev_fallbacks(self) -> None:
        settings = Settings(api_key="", audit_hmac_secret="", stream_auth_secret="")
        audit_secret, stream_secret = resolve_runtime_secrets(settings)
        self.assertEqual(audit_secret, DEV_AUDIT_SECRET)
        self.assertEqual(stream_secret, DEV_STREAM_SECRET)


class HTTPAuthTest(unittest.TestCase):
    def setUp(self) -> None:
        self.app, self.client = _make_secured_app()

    def test_protected_route_requires_api_key(self) -> None:
        response = self.client.get("/sessions")
        self.assertEqual(response.status_code, 401)

    def test_protected_route_accepts_correct_api_key(self) -> None:
        response = self.client.get("/sessions", headers={"X-API-Key": "test123"})
        self.assertEqual(response.status_code, 200)

    def test_options_preflight_bypasses_api_key(self) -> None:
        response = self.client.options(
            "/sessions",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertEqual(response.status_code, 200)

    def test_unauthorized_response_still_carries_cors_headers(self) -> None:
        response = self.client.get(
            "/sessions",
            headers={"Origin": "http://localhost:5173"},
        )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://localhost:5173")

    def test_docs_disabled_when_api_key_set(self) -> None:
        self.assertIsNone(self.app.docs_url)
        self.assertIsNone(self.app.redoc_url)
        self.assertIsNone(self.app.openapi_url)

    def test_supervisor_ws_rejects_wrong_key(self) -> None:
        with self.assertRaises(Exception):
            with self.client.websocket_connect("/ws/supervisors?api_key=wrong"):
                pass
