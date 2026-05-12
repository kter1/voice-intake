from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from voice_intake.db.base import Base
from voice_intake.models import utc_now


def _uid() -> str:
    return str(uuid4())


class KnowledgeSourceORM(Base):
    """Tracks uploaded practice documents for listing and deletion."""

    __tablename__ = "knowledge_sources"

    source_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uid)
    practice_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    category: Mapped[str] = mapped_column(String, nullable=False)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class PatientRecordORM(Base):
    """Practice-side patient roster for identity pre-population."""

    __tablename__ = "patient_records"

    patient_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uid)
    first_name: Mapped[str] = mapped_column(String, nullable=False)
    last_name: Mapped[str] = mapped_column(String, nullable=False)
    dob: Mapped[str] = mapped_column(String, nullable=False)
    member_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    insurance_plan: Mapped[str | None] = mapped_column(String, nullable=True)
    phone_normalized: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    last_visit_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    preferred_language: Mapped[str] = mapped_column(String, nullable=False, default="en")
