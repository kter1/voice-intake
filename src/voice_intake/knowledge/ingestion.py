from __future__ import annotations

from pathlib import Path
from uuid import uuid4


def chunk_document(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split text into overlapping word-count chunks."""
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += step
    return chunks


def extract_text(path: str | Path) -> str:
    """Extract plain text from PDF, DOCX, or TXT files."""
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".txt":
        return p.read_text(encoding="utf-8", errors="replace")

    if suffix == ".pdf":
        import pypdf  # type: ignore[import]

        reader = pypdf.PdfReader(str(p))
        return "\n".join(
            page.extract_text() or "" for page in reader.pages
        )

    if suffix in {".docx", ".docm"}:
        import docx  # type: ignore[import]

        doc = docx.Document(str(p))
        return "\n".join(para.text for para in doc.paragraphs)

    raise ValueError(f"Unsupported file type: {suffix!r}")


def ingest_file(
    path: str | Path,
    practice_id: str,
    category: str,
    chroma_client: object,
    source_id: str | None = None,
    chunk_size: int = 512,
    overlap: int = 64,
    embed_fn: object = None,
    collection_name: str = "practice_docs",
) -> tuple[str, int]:
    """
    Chunk, embed, and store a document in ChromaDB.

    embed_fn: optional callable(list[str]) -> list[list[float]] - injected in tests
              to avoid loading the sentence-transformers model.
    Returns (source_id, chunk_count).
    """
    p = Path(path)
    source_id = source_id or str(uuid4())
    text = extract_text(p)
    chunks = chunk_document(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return source_id, 0

    if embed_fn is not None:
        embeddings = embed_fn(chunks)
    else:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(chunks, show_progress_bar=False).tolist()

    collection = chroma_client.get_or_create_collection(collection_name)
    ids = [f"{source_id}_{i}" for i in range(len(chunks))]
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=[
            {"source_id": source_id, "practice_id": practice_id, "category": category}
        ]
        * len(chunks),
    )
    return source_id, len(chunks)
