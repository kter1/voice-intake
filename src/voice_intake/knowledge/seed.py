from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from voice_intake.knowledge.ingestion import ingest_file


@dataclass(frozen=True, slots=True)
class SeededSource:
    path: Path
    source_id: str
    chunk_count: int
    skipped: bool


def deterministic_source_id(path: str | Path, seed_root: str | Path) -> str:
    file_path = Path(path)
    root = Path(seed_root)
    relative = file_path.relative_to(root).as_posix()
    digest = hashlib.sha256()
    digest.update(relative.encode("utf-8"))
    digest.update(b"\0")
    digest.update(file_path.read_bytes())
    return f"seed_{digest.hexdigest()[:24]}"


def source_exists(
    chroma_client: object,
    source_id: str,
    collection_name: str = "practice_docs",
) -> bool:
    collection = chroma_client.get_or_create_collection(collection_name)
    try:
        result = collection.get(
            where={"source_id": source_id},
            limit=1,
            include=["metadatas"],
        )
    except Exception:
        return False
    return bool(result.get("ids", []))


def seed_directory(
    seed_root: str | Path,
    practice_id: str,
    category: str,
    chroma_client: object,
    *,
    embed_fn: object = None,
    collection_name: str = "practice_docs",
) -> list[SeededSource]:
    root = Path(seed_root)
    results: list[SeededSource] = []
    for path in sorted(root.rglob("*.txt")):
        source_id = deterministic_source_id(path, root)
        if source_exists(chroma_client, source_id, collection_name=collection_name):
            results.append(
                SeededSource(
                    path=path,
                    source_id=source_id,
                    chunk_count=0,
                    skipped=True,
                )
            )
            continue
        _, chunk_count = ingest_file(
            path=path,
            practice_id=practice_id,
            category=category,
            chroma_client=chroma_client,
            source_id=source_id,
            embed_fn=embed_fn,
            collection_name=collection_name,
        )
        results.append(
            SeededSource(
                path=path,
                source_id=source_id,
                chunk_count=chunk_count,
                skipped=False,
            )
        )
    return results
