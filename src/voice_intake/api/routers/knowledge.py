from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.orm import Session, sessionmaker

from voice_intake.api.deps import get_store
from voice_intake.db import SQLAuditStore
from voice_intake.db.base import Base
from voice_intake.knowledge.ingestion import ingest_file
from voice_intake.knowledge.orm import KnowledgeSourceORM
from voice_intake.models import utc_now

router = APIRouter(tags=["knowledge"], prefix="/knowledge")

ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt"}


class KnowledgeSourceOut(BaseModel):
    source_id: str
    practice_id: str
    category: str
    filename: str
    chunk_count: int


class KnowledgeSourceList(BaseModel):
    sources: list[KnowledgeSourceOut]


class IngestResponse(BaseModel):
    source_id: str
    chunk_count: int


@router.post("/upload", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile,
    practice_id: str = "default",
    category: str = "general",
    request: Request = None,  # type: ignore[assignment]
    store: SQLAuditStore = Depends(get_store),
) -> IngestResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type {suffix!r}. Allowed: {sorted(ALLOWED_SUFFIXES)}",
        )

    chroma_client = getattr(request.app.state, "chroma_client", None)
    if chroma_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge store not initialized.",
        )

    source_id = str(uuid4())
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    embed_fn = getattr(request.app.state, "embed_fn", None)
    try:
        _, chunk_count = ingest_file(
            path=tmp_path,
            practice_id=practice_id,
            category=category,
            chroma_client=chroma_client,
            source_id=source_id,
            embed_fn=embed_fn,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Record the source in SQLite for listing/deletion
    session_factory: sessionmaker[Session] = request.app.state.session_factory
    with session_factory() as db:
        db.add(
            KnowledgeSourceORM(
                source_id=source_id,
                practice_id=practice_id,
                category=category,
                filename=file.filename or "unknown",
                chunk_count=chunk_count,
                created_at=utc_now(),
            )
        )
        db.commit()

    return IngestResponse(source_id=source_id, chunk_count=chunk_count)


@router.get("/sources", response_model=KnowledgeSourceList)
def list_sources(request: Request, store: SQLAuditStore = Depends(get_store)) -> KnowledgeSourceList:
    session_factory: sessionmaker[Session] = request.app.state.session_factory
    with session_factory() as db:
        rows = db.query(KnowledgeSourceORM).order_by(KnowledgeSourceORM.created_at.desc()).all()
    return KnowledgeSourceList(
        sources=[
            KnowledgeSourceOut(
                source_id=r.source_id,
                practice_id=r.practice_id,
                category=r.category,
                filename=r.filename,
                chunk_count=r.chunk_count,
            )
            for r in rows
        ]
    )


@router.delete("/source/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_source(
    source_id: str,
    request: Request,
    store: SQLAuditStore = Depends(get_store),
) -> None:
    session_factory: sessionmaker[Session] = request.app.state.session_factory
    chroma_client = getattr(request.app.state, "chroma_client", None)

    with session_factory() as db:
        row = db.get(KnowledgeSourceORM, source_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"Source {source_id!r} not found")
        db.delete(row)
        db.commit()

    if chroma_client is not None:
        try:
            collection = chroma_client.get_or_create_collection("practice_docs")
            collection.delete(where={"source_id": source_id})
        except Exception:
            pass
