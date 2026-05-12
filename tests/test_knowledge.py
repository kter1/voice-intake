"""
Tests for the Knowledge & RAG layer (Phase 4).
Uses ChromaDB EphemeralClient (in-memory) and mock embeddings - no model download required.

NOTE: chromadb.EphemeralClient() instances share a process-level Rust server. Tests that
add embeddings must use unique collection names to avoid cross-test dimension conflicts.
"""
import unittest
from pathlib import Path
import tempfile
import uuid
from unittest.mock import MagicMock, patch


def _mock_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic mock embeddings: 4-dim, based on text length."""
    return [
        [float(len(t) % 10) / 10.0, float(len(t) % 5) / 5.0,
         float(len(t) % 3) / 3.0, float(len(t) % 7) / 7.0]
        for t in texts
    ]


def _col() -> str:
    """Unique collection name per call to avoid shared-state dimension conflicts."""
    return f"test_{uuid.uuid4().hex}"


class ChunkDocumentTest(unittest.TestCase):
    def test_empty_text_returns_empty_list(self):
        from voice_intake.knowledge.ingestion import chunk_document
        self.assertEqual(chunk_document(""), [])

    def test_short_text_returns_single_chunk(self):
        from voice_intake.knowledge.ingestion import chunk_document
        text = "hello world this is a short doc"
        chunks = chunk_document(text, chunk_size=100, overlap=10)
        self.assertEqual(len(chunks), 1)
        self.assertIn("hello", chunks[0])

    def test_long_text_returns_multiple_chunks(self):
        from voice_intake.knowledge.ingestion import chunk_document
        text = " ".join([f"word{i}" for i in range(600)])
        chunks = chunk_document(text, chunk_size=100, overlap=20)
        self.assertGreater(len(chunks), 1)

    def test_overlap_causes_word_repetition(self):
        from voice_intake.knowledge.ingestion import chunk_document
        text = " ".join([str(i) for i in range(200)])
        chunks = chunk_document(text, chunk_size=50, overlap=10)
        end_words = set(chunks[0].split()[-10:])
        start_words = set(chunks[1].split()[:10])
        self.assertTrue(end_words & start_words)


class ExtractTextTest(unittest.TestCase):
    def test_extract_txt(self):
        from voice_intake.knowledge.ingestion import extract_text
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("Practice policy: appointments are 30 minutes.")
            path = f.name
        try:
            text = extract_text(path)
            self.assertIn("appointments", text)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_unsupported_extension_raises(self):
        from voice_intake.knowledge.ingestion import extract_text
        with self.assertRaises(ValueError):
            extract_text("fake.mp3")


class KnowledgeRetrieverTest(unittest.TestCase):
    def test_constructs_model_eagerly_when_no_embed_fn(self):
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_instance = MagicMock()
            mock_st.return_value = mock_instance

            from voice_intake.knowledge.retrieval import KnowledgeRetriever

            retriever = KnowledgeRetriever(MagicMock(), embed_fn=None, collection_name=_col())

        mock_st.assert_called_once_with("all-MiniLM-L6-v2")
        self.assertIs(retriever._model, mock_instance)

    def test_skips_model_construction_when_embed_fn_provided(self):
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            from voice_intake.knowledge.retrieval import KnowledgeRetriever

            retriever = KnowledgeRetriever(
                MagicMock(),
                embed_fn=lambda texts: [[0.0] for _ in texts],
                collection_name=_col(),
            )

        mock_st.assert_not_called()
        self.assertIsNone(retriever._model)

    def test_retrieve_returns_empty_when_no_docs(self):
        import chromadb
        from voice_intake.knowledge.retrieval import KnowledgeRetriever
        client = chromadb.EphemeralClient()
        retriever = KnowledgeRetriever(client, embed_fn=_mock_embed, collection_name=_col())
        results = retriever.retrieve("appointment scheduling")
        self.assertEqual(results, [])

    def test_retrieve_returns_chunks_after_ingest(self):
        import chromadb
        from voice_intake.knowledge.retrieval import KnowledgeChunk, KnowledgeRetriever

        col = _col()
        client = chromadb.EphemeralClient()
        collection = client.get_or_create_collection(col)
        collection.add(
            ids=["src1_0", "src1_1"],
            embeddings=_mock_embed(["appointment policy", "cancellation fee"]),
            documents=["Appointments are 30 minutes.", "Cancellation requires 24-hour notice."],
            metadatas=[
                {"source_id": "src1", "practice_id": "p1", "category": "policy"},
                {"source_id": "src1", "practice_id": "p1", "category": "policy"},
            ],
        )
        retriever = KnowledgeRetriever(client, embed_fn=_mock_embed, collection_name=col)
        results = retriever.retrieve("appointments", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertIsInstance(results[0], KnowledgeChunk)
        self.assertIsNotNone(results[0].text)

    def test_retrieve_returns_at_most_top_k(self):
        import chromadb
        from voice_intake.knowledge.retrieval import KnowledgeRetriever

        col = _col()
        client = chromadb.EphemeralClient()
        collection = client.get_or_create_collection(col)
        collection.add(
            ids=[f"s_{i}" for i in range(5)],
            embeddings=_mock_embed([f"document {i}" for i in range(5)]),
            documents=[f"Document {i} content" for i in range(5)],
            metadatas=[{"source_id": "s", "practice_id": "p", "category": "x"}] * 5,
        )
        retriever = KnowledgeRetriever(client, embed_fn=_mock_embed, collection_name=col)
        results = retriever.retrieve("query", top_k=3)
        self.assertLessEqual(len(results), 3)


class EligibilityTest(unittest.TestCase):
    def test_mock_adapter_returns_eligible(self):
        from voice_intake.knowledge.eligibility import MockEligibilityAdapter
        adapter = MockEligibilityAdapter()
        result = adapter.check_eligibility("MEM001", "1980-01-01", "BlueCross")
        self.assertTrue(result.eligible)
        self.assertEqual(result.plan_name, "BlueCross")
        self.assertIsNotNone(result.copay)

    def test_mock_adapter_satisfies_protocol(self):
        from voice_intake.knowledge.eligibility import EligibilityInterface, MockEligibilityAdapter
        adapter = MockEligibilityAdapter()
        self.assertIsInstance(adapter, EligibilityInterface)

    def test_availity_adapter_raises_not_implemented(self):
        from voice_intake.knowledge.eligibility import AvailityAdapter
        adapter = AvailityAdapter("client_id", "secret")
        with self.assertRaises(NotImplementedError):
            adapter.check_eligibility("MEM001", "1980-01-01", "BlueCross")


class KnowledgeAPITest(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from voice_intake.api.app import create_app
        from voice_intake.api.deps import get_store
        from voice_intake.config import Settings
        from voice_intake.db import SQLAuditStore, create_tables, get_engine, make_session_factory
        import chromadb

        test_settings = Settings(
            database_url="sqlite:///:memory:",
            mock_llm=True,
            audit_hmac_secret="test-audit-secret",
            stream_auth_secret="test-stream-secret",
        )
        app = create_app(settings=test_settings)
        engine = get_engine("sqlite:///:memory:")
        create_tables(engine)
        factory = make_session_factory(engine)
        store = SQLAuditStore(factory, hmac_secret="test-audit-secret")
        app.dependency_overrides[get_store] = lambda: store
        app.state.session_factory = factory
        app.state.chroma_client = chromadb.EphemeralClient()
        # Use mock embed so tests don't load the real sentence-transformers model
        # and don't pollute the shared "practice_docs" collection with 384-dim embeddings
        app.state.embed_fn = _mock_embed
        self.client = TestClient(app, raise_server_exceptions=True)

    def test_list_sources_returns_empty_initially(self):
        resp = self.client.get("/knowledge/sources")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["sources"], [])

    def test_upload_txt_document(self):
        content = b"This is a practice policy document about appointments and scheduling."
        resp = self.client.post(
            "/knowledge/upload?practice_id=p1&category=policy",
            files={"file": ("policy.txt", content, "text/plain")},
        )
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertIn("source_id", data)
        self.assertGreater(data["chunk_count"], 0)

    def test_upload_unsupported_type_returns_422(self):
        resp = self.client.post(
            "/knowledge/upload",
            files={"file": ("audio.mp3", b"fake", "audio/mpeg")},
        )
        self.assertEqual(resp.status_code, 422)

    def test_uploaded_source_appears_in_list(self):
        content = b"Policy: no same-day cancellations without a fee."
        self.client.post(
            "/knowledge/upload?practice_id=p1&category=policy",
            files={"file": ("rules.txt", content, "text/plain")},
        )
        resp = self.client.get("/knowledge/sources")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.json()["sources"]), 1)
        self.assertEqual(resp.json()["sources"][0]["filename"], "rules.txt")

    def test_delete_source_removes_it(self):
        content = b"Provider directory for primary care."
        upload_resp = self.client.post(
            "/knowledge/upload?practice_id=p1&category=providers",
            files={"file": ("providers.txt", content, "text/plain")},
        )
        source_id = upload_resp.json()["source_id"]
        del_resp = self.client.delete(f"/knowledge/source/{source_id}")
        self.assertEqual(del_resp.status_code, 204)
        list_resp = self.client.get("/knowledge/sources")
        self.assertEqual(len(list_resp.json()["sources"]), 0)

    def test_delete_nonexistent_source_returns_404(self):
        resp = self.client.delete("/knowledge/source/does-not-exist")
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
