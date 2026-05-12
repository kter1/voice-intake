from __future__ import annotations

import re
import uuid
from pathlib import Path


SEED_ROOT = Path("data/rag_seed/demo_practice")


def _collection_name() -> str:
    return f"seed_{uuid.uuid4().hex}"


def _seed_embed(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        lower = text.lower()
        vectors.append(
            [
                float(lower.count("appointment") + lower.count("scheduling")),
                float(lower.count("insurance") + lower.count("payer")),
                float(lower.count("emergency") + lower.count("scope")),
                float(len(text) % 11) / 11.0,
            ]
        )
    return vectors


def test_seed_files_contain_no_obvious_phi_patterns():
    joined = "\n".join(path.read_text(encoding="utf-8") for path in SEED_ROOT.glob("*.txt"))

    assert not re.search(r"\b\d{8,}\b", joined)
    assert not re.search(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){2}\d{4}\b", joined)
    assert "patient records" in joined
    assert "synthetic demo content" in joined


def test_deterministic_source_id_is_stable():
    from voice_intake.knowledge.seed import deterministic_source_id

    path = SEED_ROOT / "appointments.txt"

    assert deterministic_source_id(path, SEED_ROOT) == deterministic_source_id(path, SEED_ROOT)
    assert deterministic_source_id(path, SEED_ROOT).startswith("seed_")


def test_repeated_seeding_does_not_duplicate_chunks():
    import chromadb
    from voice_intake.knowledge.seed import seed_directory

    collection_name = _collection_name()
    client = chromadb.EphemeralClient()

    first = seed_directory(
        SEED_ROOT,
        practice_id="demo",
        category="seed",
        chroma_client=client,
        embed_fn=_seed_embed,
        collection_name=collection_name,
    )
    collection = client.get_or_create_collection(collection_name)
    first_count = collection.count()
    second = seed_directory(
        SEED_ROOT,
        practice_id="demo",
        category="seed",
        chroma_client=client,
        embed_fn=_seed_embed,
        collection_name=collection_name,
    )

    assert first_count > 0
    assert collection.count() == first_count
    assert all(not result.skipped for result in first)
    assert all(result.skipped for result in second)


def test_retrieval_returns_seeded_policy_content():
    import chromadb
    from voice_intake.knowledge.retrieval import KnowledgeRetriever
    from voice_intake.knowledge.seed import seed_directory

    collection_name = _collection_name()
    client = chromadb.EphemeralClient()
    seed_directory(
        SEED_ROOT,
        practice_id="demo",
        category="seed",
        chroma_client=client,
        embed_fn=_seed_embed,
        collection_name=collection_name,
    )
    retriever = KnowledgeRetriever(
        client,
        embed_fn=_seed_embed,
        collection_name=collection_name,
    )

    chunks = retriever.retrieve("appointment scheduling", practice_id="demo", top_k=3)

    assert chunks
    assert any("appointment" in chunk.text.lower() for chunk in chunks)
    assert all(chunk.practice_id == "demo" for chunk in chunks)


def test_generated_local_rag_paths_are_ignored():
    gitignore = Path(".gitignore").read_text(encoding="utf-8")

    assert "chroma_data/" in gitignore
    assert ".chroma-demo/" in gitignore
    assert "uploads/" in gitignore
