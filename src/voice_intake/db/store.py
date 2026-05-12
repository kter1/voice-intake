from __future__ import annotations

from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import IntegrityError

from voice_intake.models import (
    AppointmentRequest,
    AuditEvent,
    CallSession,
    ConsentArtifact,
    ModelProposal,
    PolicyDecision,
    SupervisorIntervention,
    ValidatorResult,
    VoiceTurn,
    utc_now,
)

from .audit_canonical import compute_chain_hash
from .models_orm import (
    AppointmentRequestORM,
    AuditEventORM,
    ConsentArtifactORM,
    ModelProposalORM,
    PolicyDecisionORM,
    SessionORM,
    SupervisorInterventionORM,
    ValidatorResultORM,
    VoiceTurnORM,
)


class SQLAuditStore:
    def __init__(self, session_factory: sessionmaker[Session], *, hmac_secret: str) -> None:
        self._factory = session_factory
        self._secret = hmac_secret.encode()

    @property
    def audit_secret(self) -> str:
        return self._secret.decode()

    # ── AuditStore Protocol methods (mirror InMemoryAuditStore) ──────────────

    def record_proposal(self, proposal: ModelProposal) -> None:
        with self._session() as db:
            row = ModelProposalORM.from_domain(
                session_id=proposal.session_id or "",
                proposal=proposal,
            )
            db.merge(row)
            db.commit()

    def record_validator_result(self, result: ValidatorResult) -> None:
        with self._session() as db:
            db.merge(ValidatorResultORM.from_domain(result))
            db.commit()

    def record_policy_decision(self, decision: object) -> None:
        if not isinstance(decision, PolicyDecision):
            return
        with self._session() as db:
            db.merge(PolicyDecisionORM.from_domain(decision))
            db.commit()

    def record_audit_event(self, event: AuditEvent) -> None:
        attempts = 0
        while True:
            attempts += 1
            with self._session() as db:
                try:
                    previous_hash, next_index = self._latest_chain_tip(db, event.session_id)
                    event.chain_index = next_index
                    event.chain_hash = compute_chain_hash(self._secret, previous_hash, event)
                    db.add(AuditEventORM.from_domain(event))
                    db.commit()
                    return
                except IntegrityError:
                    db.rollback()
                    if attempts >= 3:
                        raise

    def record_consent_artifact(self, artifact: ConsentArtifact) -> None:
        with self._session() as db:
            db.add(ConsentArtifactORM.from_domain(artifact))
            db.commit()

    def record_supervisor_intervention(self, intervention: SupervisorIntervention) -> None:
        with self._session() as db:
            db.add(SupervisorInterventionORM.from_domain(intervention))
            db.commit()

    def consents_for_session(self, session_id: str) -> list[ConsentArtifact]:
        with self._session() as db:
            rows = (
                db.query(ConsentArtifactORM)
                .filter(ConsentArtifactORM.session_id == session_id)
                .all()
            )
            return [r.to_domain() for r in rows]

    # ── Extra methods needed by the API layer ─────────────────────────────────

    def save_session(
        self,
        session: CallSession,
        reject_counts: dict | None = None,
        caller_phone: str | None = None,
    ) -> None:
        with self._session() as db:
            existing = db.get(SessionORM, session.session_id)
            if existing:
                existing.current_state = session.current_state.value
                existing.mode = session.mode.value
                existing.ended_at = session.ended_at
                existing.disposition = (
                    session.disposition.value if session.disposition else None
                )
                existing.metadata_json = __import__("json").dumps(session.metadata)
                if reject_counts is not None:
                    existing.set_reject_counts(reject_counts)
                if caller_phone is not None:
                    existing.caller_phone_normalized = caller_phone
            else:
                db.add(SessionORM.from_domain(session, reject_counts, caller_phone))
            db.commit()

    def load_session(self, session_id: str) -> CallSession | None:
        with self._session() as db:
            row = db.get(SessionORM, session_id)
            return row.to_domain() if row else None

    def load_reject_counts(self, session_id: str) -> dict[tuple[str, str], int]:
        with self._session() as db:
            row = db.get(SessionORM, session_id)
            return row.reject_counts() if row else {}

    def audit_events_for_session(self, session_id: str) -> list[AuditEvent]:
        with self._session() as db:
            rows = (
                db.query(AuditEventORM)
                .filter(AuditEventORM.session_id == session_id)
                .order_by(
                    AuditEventORM.chain_index.is_(None),
                    AuditEventORM.chain_index.asc(),
                    AuditEventORM.timestamp.asc(),
                )
                .all()
            )
            return [r.to_domain() for r in rows]

    def turns_for_session(self, session_id: str) -> list[VoiceTurn]:
        with self._session() as db:
            rows = (
                db.query(VoiceTurnORM)
                .filter(VoiceTurnORM.session_id == session_id)
                .all()
            )
            return [r.to_domain() for r in rows]

    def record_turn(self, session_id: str, turn: VoiceTurn) -> None:
        with self._session() as db:
            db.merge(VoiceTurnORM.from_domain(session_id, turn))
            db.commit()

    def list_sessions(self) -> list[CallSession]:
        with self._session() as db:
            rows = db.query(SessionORM).order_by(SessionORM.started_at.desc()).all()
            return [r.to_domain() for r in rows]

    # ── Appointment requests (demo scheduling flow) ───────────────────────────

    def get_appointment_request(self, session_id: str) -> AppointmentRequest | None:
        """Return the appointment request for a session, or None if not yet created."""
        with self._session() as db:
            row = (
                db.query(AppointmentRequestORM)
                .filter(AppointmentRequestORM.session_id == session_id)
                .first()
            )
            return row.to_domain() if row else None

    def upsert_appointment_request(self, request: AppointmentRequest) -> None:
        """Insert or update the appointment request for a session (one row per session)."""
        with self._session() as db:
            existing = (
                db.query(AppointmentRequestORM)
                .filter(AppointmentRequestORM.session_id == request.session_id)
                .first()
            )
            if existing is not None:
                existing.reason_for_visit = request.reason_for_visit
                existing.patient_name = request.patient_name
                existing.date_of_birth = request.date_of_birth
                existing.insurance_name = request.insurance_name
                existing.insurance_network_status = request.insurance_network_status.value
                existing.appointment_status = request.appointment_status.value
                existing.scheduled_slot = request.scheduled_slot
                existing.updated_at = utc_now()
            else:
                db.add(AppointmentRequestORM.from_domain(request))
            db.commit()

    # ─────────────────────────────────────────────────────────────────────────

    def _session(self) -> Session:
        return self._factory()

    def _latest_chain_tip(self, db: Session, session_id: str) -> tuple[str, int]:
        row = (
            db.query(AuditEventORM.chain_hash, AuditEventORM.chain_index)
            .filter(
                AuditEventORM.session_id == session_id,
                AuditEventORM.chain_index.is_not(None),
            )
            .order_by(AuditEventORM.chain_index.desc())
            .first()
        )
        if row is None:
            return "", 0
        chain_hash, chain_index = row
        return chain_hash or "", int(chain_index) + 1
