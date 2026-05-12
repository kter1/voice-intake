from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from .orm import PatientRecordORM


@dataclass
class PatientRecord:
    patient_id: str
    first_name: str
    last_name: str
    dob: str
    member_id: str | None
    insurance_plan: str | None
    phone_normalized: str | None
    last_visit_at: datetime | None
    preferred_language: str


class PatientLookupService:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._factory = session_factory

    def lookup_patient(
        self, phone_normalized: str, member_id: str | None = None
    ) -> PatientRecord | None:
        with self._factory() as db:
            q = db.query(PatientRecordORM)
            if member_id:
                row = (
                    q.filter(PatientRecordORM.member_id == member_id).first()
                    or q.filter(
                        PatientRecordORM.phone_normalized == phone_normalized
                    ).first()
                )
            else:
                row = q.filter(
                    PatientRecordORM.phone_normalized == phone_normalized
                ).first()
            if row is None:
                return None
            return PatientRecord(
                patient_id=row.patient_id,
                first_name=row.first_name,
                last_name=row.last_name,
                dob=row.dob,
                member_id=row.member_id,
                insurance_plan=row.insurance_plan,
                phone_normalized=row.phone_normalized,
                last_visit_at=row.last_visit_at,
                preferred_language=row.preferred_language,
            )

    def seed_from_csv(self, csv_path: str | Path) -> int:
        """Load patient records from CSV for MOCK_PATIENT_LOOKUP mode."""
        rows_added = 0
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            with self._factory() as db:
                for row in reader:
                    db.merge(
                        PatientRecordORM(
                            patient_id=row.get("patient_id", ""),
                            first_name=row.get("first_name", ""),
                            last_name=row.get("last_name", ""),
                            dob=row.get("dob", ""),
                            member_id=row.get("member_id") or None,
                            insurance_plan=row.get("insurance_plan") or None,
                            phone_normalized=row.get("phone_normalized") or None,
                            preferred_language=row.get("preferred_language", "en"),
                        )
                    )
                    rows_added += 1
                db.commit()
        return rows_added
