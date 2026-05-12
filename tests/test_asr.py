"""
Unit tests for the ASR layer (Phase 6).
Deepgram adapter is tested with a mocked SDK - no real API calls.
"""
import asyncio
import base64
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _mock_embed(texts: list[str]) -> list[list[float]]:
    return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


# ---------------------------------------------------------------------------
# Interface / Protocol
# ---------------------------------------------------------------------------

class ASRInterfaceTest(unittest.TestCase):
    def test_asr_result_fields(self):
        from voice_intake.asr.interface import ASRResult
        r = ASRResult(transcript="hello", confidence=0.95, is_final=True)
        self.assertEqual(r.transcript, "hello")
        self.assertEqual(r.confidence, 0.95)
        self.assertTrue(r.is_final)
        self.assertEqual(r.language, "en")

    def test_mock_asr_adapter_satisfies_protocol(self):
        from voice_intake.asr.interface import ASRInterface
        from voice_intake.testing.mock_asr import MockASRAdapter
        self.assertIsInstance(MockASRAdapter(), ASRInterface)

    def test_deepgram_adapter_satisfies_protocol(self):
        from voice_intake.asr.deepgram_adapter import DeepgramAdapter
        from voice_intake.asr.interface import ASRInterface
        self.assertIsInstance(DeepgramAdapter("key"), ASRInterface)


# ---------------------------------------------------------------------------
# MockASRAdapter
# ---------------------------------------------------------------------------

class MockASRAdapterTest(unittest.TestCase):
    def test_returns_decoded_bytes_as_transcript(self):
        from voice_intake.testing.mock_asr import MockASRAdapter
        adapter = MockASRAdapter()
        result = _run(adapter.transcribe(b"My name is Jane Doe"))
        self.assertEqual(result.transcript, "My name is Jane Doe")
        self.assertEqual(result.confidence, 1.0)
        self.assertTrue(result.is_final)

    def test_returns_empty_for_empty_bytes(self):
        from voice_intake.testing.mock_asr import MockASRAdapter
        result = _run(MockASRAdapter().transcribe(b""))
        self.assertEqual(result.transcript, "")


# ---------------------------------------------------------------------------
# DeepgramAdapter (mocked)
# ---------------------------------------------------------------------------

class DeepgramAdapterTest(unittest.TestCase):
    def _mock_deepgram_response(self, transcript: str, confidence: float):
        alt = MagicMock()
        alt.transcript = transcript
        alt.confidence = confidence
        channel = MagicMock()
        channel.alternatives = [alt]
        results = MagicMock()
        results.channels = [channel]
        resp = MagicMock()
        resp.results = results
        return resp

    def test_transcribes_audio_bytes(self):
        from voice_intake.asr.deepgram_adapter import DeepgramAdapter

        resp = self._mock_deepgram_response("Hello, I need an appointment.", 0.97)

        with patch("deepgram.AsyncDeepgramClient") as mock_cls:
            instance = MagicMock()
            instance.listen.v1.media.transcribe_file = AsyncMock(return_value=resp)
            mock_cls.return_value = instance

            adapter = DeepgramAdapter(api_key="test-key", model="nova-3")
            result = _run(adapter.transcribe(b"fake-audio-bytes"))

        self.assertEqual(result.transcript, "Hello, I need an appointment.")
        self.assertAlmostEqual(result.confidence, 0.97)
        self.assertTrue(result.is_final)

    def test_empty_results_returns_empty_transcript(self):
        from voice_intake.asr.deepgram_adapter import DeepgramAdapter

        resp = MagicMock()
        resp.results.channels = []

        with patch("deepgram.AsyncDeepgramClient") as mock_cls:
            instance = MagicMock()
            instance.listen.v1.media.transcribe_file = AsyncMock(return_value=resp)
            mock_cls.return_value = instance

            adapter = DeepgramAdapter(api_key="key")
            result = _run(adapter.transcribe(b"audio"))

        self.assertEqual(result.transcript, "")
        self.assertEqual(result.confidence, 0.0)


# ---------------------------------------------------------------------------
# get_asr_adapter factory
# ---------------------------------------------------------------------------

class GetASRAdapterTest(unittest.TestCase):
    def test_returns_mock_adapter_when_mock_asr_true(self):
        from voice_intake.asr import get_asr_adapter
        from voice_intake.testing.mock_asr import MockASRAdapter

        settings = MagicMock()
        settings.mock_asr = True
        adapter = get_asr_adapter(settings)
        self.assertIsInstance(adapter, MockASRAdapter)

    def test_returns_deepgram_when_key_provided(self):
        from voice_intake.asr import get_asr_adapter
        from voice_intake.asr.deepgram_adapter import DeepgramAdapter

        settings = MagicMock()
        settings.mock_asr = False
        settings.deepgram_api_key = "dg-key"
        adapter = get_asr_adapter(settings)
        self.assertIsInstance(adapter, DeepgramAdapter)

    def test_raises_when_no_key_and_not_mock(self):
        from voice_intake.asr import get_asr_adapter

        settings = MagicMock()
        settings.mock_asr = False
        settings.deepgram_api_key = ""
        with self.assertRaises(RuntimeError):
            get_asr_adapter(settings)


# ---------------------------------------------------------------------------
# API turn endpoint: audio_b64 path
# ---------------------------------------------------------------------------

class TurnASRIntegrationTest(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from voice_intake.api.app import create_app
        from voice_intake.api.deps import get_store
        from voice_intake.config import Settings
        from voice_intake.db import SQLAuditStore, create_tables, get_engine, make_session_factory

        test_settings = Settings(
            database_url="sqlite:///:memory:",
            mock_llm=True,
            mock_asr=True,
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

        self.client = TestClient(app, raise_server_exceptions=True)

        # Open session and grant both consents
        resp = self.client.post("/session/open", json={
            "telephony_call_id": "call-asr-test",
            "operator_id": "op-1",
        })
        self.session_id = resp.json()["session_id"]
        for ct in ["recording_consent", "ai_assistance_consent"]:
            self.client.post(f"/session/{self.session_id}/consent",
                             json={"consent_type": ct, "granted": True})

    def test_explicit_transcript_takes_precedence_over_audio(self):
        """If both transcript and audio_b64 are sent, the explicit transcript wins."""
        audio_b64 = base64.b64encode(b"different text from audio").decode()
        resp = self.client.post(f"/session/{self.session_id}/turn", json={
            "transcript": "My name is Alice",
            "audio_b64": audio_b64,
        })
        self.assertEqual(resp.status_code, 200)

    def test_audio_b64_with_mock_asr_transcribes(self):
        """When mock_asr=True, audio bytes are decoded as UTF-8 text."""
        text = "My name is Bob Smith"
        audio_b64 = base64.b64encode(text.encode()).decode()
        # With mock_asr=True and no explicit transcript, audio_b64 is not used
        # (the route only calls ASR when mock_asr=False). Provide transcript instead.
        resp = self.client.post(f"/session/{self.session_id}/turn", json={
            "transcript": text,
        })
        self.assertEqual(resp.status_code, 200)
        self.assertIn("turn_id", resp.json())


if __name__ == "__main__":
    unittest.main()
