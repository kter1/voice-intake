from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class KnowledgeChunk:
    text: str
    source: str
    practice_id: str
    category: str
    score: float


class KnowledgeRetriever:
    """
    Semantic retrieval over practice documents stored in ChromaDB.

    Accepts an optional embed_fn to allow injection of mock embeddings in tests
    without loading the full sentence-transformers model.
    """

    def __init__(
        self,
        chroma_client: object,
        model_name: str = "all-MiniLM-L6-v2",
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        collection_name: str = "practice_docs",
    ) -> None:
        self._client = chroma_client
        self._model_name = model_name
        self._embed_fn = embed_fn
        self._collection_name = collection_name
        if embed_fn is None:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]

            self._model: object | None = SentenceTransformer(model_name)
        else:
            self._model = None

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._embed_fn is not None:
            return self._embed_fn(texts)
        return self._model.encode(texts, show_progress_bar=False).tolist()  # type: ignore[union-attr]

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        practice_id: str | None = None,
    ) -> list[KnowledgeChunk]:
        try:
            collection = self._client.get_or_create_collection(self._collection_name)
        except Exception:
            return []

        query_embedding = self._embed([query])[0]
        where: dict | None = {"practice_id": practice_id} if practice_id else None
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, max(1, collection.count())),
                include=["documents", "metadatas", "distances"],
                **({"where": where} if where else {}),
            )
        except Exception:
            return []

        chunks: list[KnowledgeChunk] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            chunks.append(
                KnowledgeChunk(
                    text=doc,
                    source=meta.get("source_id", ""),
                    practice_id=meta.get("practice_id", ""),
                    category=meta.get("category", ""),
                    score=float(1.0 - dist),
                )
            )
        return chunks
