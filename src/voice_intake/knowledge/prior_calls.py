from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session, sessionmaker

from voice_intake.db.models_orm import SessionORM, VoiceTurnORM
from voice_intake.models import CallDisposition


@dataclass
class PriorCallSummary:
    session_id: str
    started_at: datetime
    disposition: CallDisposition | None
    caller_phone: str | None
    field_snapshots: dict


def get_prior_calls(
    session_factory: sessionmaker[Session],
    phone_normalized: str,
    limit: int = 3,
) -> list[PriorCallSummary]:
    """Return the last `limit` completed sessions for a given caller phone."""
    with session_factory() as db:
        rows = (
            db.query(SessionORM)
            .filter(SessionORM.caller_phone_normalized == phone_normalized)
            .filter(SessionORM.disposition.isnot(None))
            .order_by(SessionORM.started_at.desc())
            .limit(limit)
            .all()
        )
        summaries: list[PriorCallSummary] = []
        for row in rows:
            # Pull the last few turn transcripts as field snapshots
            turns = (
                db.query(VoiceTurnORM)
                .filter(VoiceTurnORM.session_id == row.session_id)
                .order_by(VoiceTurnORM.start_ts.asc())
                .limit(10)
                .all()
            )
            field_snapshots = {
                f"turn_{i}": t.transcript[:120] for i, t in enumerate(turns)
            }
            summaries.append(
                PriorCallSummary(
                    session_id=row.session_id,
                    started_at=row.started_at,
                    disposition=(
                        CallDisposition(row.disposition) if row.disposition else None
                    ),
                    caller_phone=row.caller_phone_normalized,
                    field_snapshots=field_snapshots,
                )
            )
        return summaries
