"""
Tests for patient lookup, prior call history, and prompting RAG context injection.
"""
import csv
import tempfile
import unittest
from pathlib import Path


def _make_factory():
    from voice_intake.db.base import Base, get_engine, make_session_factory
    from voice_intake.db import create_tables

    engine = get_engine("sqlite:///:memory:")
    create_tables(engine)
    return make_session_factory(engine)


class PatientLookupTest(unittest.TestCase):
    def setUp(self):
        self.factory = _make_factory()
        from voice_intake.knowledge.patient_lookup import PatientLookupService
        self.service = PatientLookupService(self.factory)

    def _seed_patient(self):
        from voice_intake.knowledge.orm import PatientRecordORM
        from voice_intake.models import utc_now

        with self.factory() as db:
            db.add(
                PatientRecordORM(
                    patient_id="pat-001",
                    first_name="Jane",
                    last_name="Doe",
                    dob="1985-03-14",
                    member_id="MEM001",
                    insurance_plan="BlueCross",
                    phone_normalized="+15550001234",
                    preferred_language="en",
                )
            )
            db.commit()

    def test_lookup_by_phone_found(self):
        self._seed_patient()
        record = self.service.lookup_patient("+15550001234")
        self.assertIsNotNone(record)
        self.assertEqual(record.first_name, "Jane")
        self.assertEqual(record.member_id, "MEM001")

    def test_lookup_by_phone_not_found(self):
        record = self.service.lookup_patient("+19999999999")
        self.assertIsNone(record)

    def test_lookup_by_member_id_preferred(self):
        self._seed_patient()
        record = self.service.lookup_patient("+10000000000", member_id="MEM001")
        self.assertIsNotNone(record)
        self.assertEqual(record.last_name, "Doe")

    def test_seed_from_csv(self):
        rows = [
            {"patient_id": "p1", "first_name": "Alice", "last_name": "Smith",
             "dob": "1990-05-01", "member_id": "M1", "insurance_plan": "Aetna",
             "phone_normalized": "+15551111111", "preferred_language": "en"},
            {"patient_id": "p2", "first_name": "Bob", "last_name": "Jones",
             "dob": "1975-12-25", "member_id": "M2", "insurance_plan": "United",
             "phone_normalized": "+15552222222", "preferred_language": "es"},
        ]
        with tempfile.NamedTemporaryFile(
            suffix=".csv", mode="w", newline="", delete=False
        ) as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
            csv_path = f.name
        try:
            count = self.service.seed_from_csv(csv_path)
            self.assertEqual(count, 2)
            record = self.service.lookup_patient("+15551111111")
            self.assertIsNotNone(record)
            self.assertEqual(record.first_name, "Alice")
        finally:
            Path(csv_path).unlink(missing_ok=True)


class PriorCallsTest(unittest.TestCase):
    def setUp(self):
        self.factory = _make_factory()

    def _open_session_with_phone(self, phone: str, session_id: str, disposition: str | None = "intake_complete"):
        from voice_intake.db.models_orm import SessionORM
        from voice_intake.models import (
            CallDisposition, CallMode, CallSession, CallState,
            StirShakenAttestation, utc_now,
        )

        session = CallSession(
            session_id=session_id,
            telephony_call_id=f"tel-{session_id}",
            operator_id="op-1",
            current_state=CallState.CLOSE,
            mode=CallMode.AI_LED,
            policy_profile_id="default",
            template_bundle_version="1.0",
            language_status="en",
            stir_shaken_attestation=StirShakenAttestation.A,
            started_at=utc_now(),
            disposition=CallDisposition(disposition) if disposition else None,
        )
        with self.factory() as db:
            row = SessionORM.from_domain(session, caller_phone=phone)
            db.add(row)
            db.commit()

    def test_returns_empty_for_unknown_phone(self):
        from voice_intake.knowledge.prior_calls import get_prior_calls
        results = get_prior_calls(self.factory, "+10000000000")
        self.assertEqual(results, [])

    def test_returns_prior_calls_for_known_phone(self):
        from voice_intake.knowledge.prior_calls import get_prior_calls
        self._open_session_with_phone("+15550001234", "s1")
        self._open_session_with_phone("+15550001234", "s2")
        results = get_prior_calls(self.factory, "+15550001234")
        self.assertEqual(len(results), 2)

    def test_respects_limit(self):
        from voice_intake.knowledge.prior_calls import get_prior_calls
        for i in range(5):
            self._open_session_with_phone("+15550001234", f"s{i}")
        results = get_prior_calls(self.factory, "+15550001234", limit=2)
        self.assertEqual(len(results), 2)

    def test_excludes_sessions_without_disposition(self):
        from voice_intake.knowledge.prior_calls import get_prior_calls
        self._open_session_with_phone("+15550001234", "complete-session", disposition="intake_complete")
        self._open_session_with_phone("+15550001234", "incomplete-session", disposition=None)
        results = get_prior_calls(self.factory, "+15550001234")
        session_ids = [r.session_id for r in results]
        self.assertIn("complete-session", session_ids)
        self.assertNotIn("incomplete-session", session_ids)


class PromptingRAGContextTest(unittest.TestCase):
    def _make_session(self):
        from voice_intake.models import (
            CallMode, CallSession, CallState, StirShakenAttestation, utc_now,
        )
        return CallSession(
            session_id="s1",
            telephony_call_id="t1",
            operator_id="op1",
            current_state=CallState.IDENTITY_CAPTURE,
            mode=CallMode.AI_LED,
            policy_profile_id="default",
            template_bundle_version="1.0",
            language_status="en",
            stir_shaken_attestation=StirShakenAttestation.A,
            started_at=utc_now(),
        )

    def test_knowledge_chunks_in_prompt(self):
        from voice_intake.prompting import build_model_prompt
        session = self._make_session()
        chunks = [{"text": "Policy doc", "source": "s1", "category": "policy", "score": 0.9}]
        prompt = build_model_prompt(session, [], [], [], knowledge_chunks=chunks)
        self.assertIn("knowledge_context", prompt)
        self.assertEqual(prompt["knowledge_context"][0]["text"], "Policy doc")

    def test_patient_context_in_prompt(self):
        from voice_intake.prompting import build_model_prompt
        session = self._make_session()
        ctx = {"first_name": "Jane", "dob": "****-03-14"}
        prompt = build_model_prompt(session, [], [], [], patient_context=ctx)
        self.assertIn("patient_context", prompt)

    def test_eligibility_context_in_prompt(self):
        from voice_intake.prompting import build_model_prompt
        session = self._make_session()
        elig = {"eligible": True, "plan_name": "BlueCross", "copay": "$20", "notes": "ok"}
        prompt = build_model_prompt(session, [], [], [], eligibility_context=elig)
        self.assertIn("eligibility_context", prompt)

    def test_prior_calls_in_prompt(self):
        from voice_intake.prompting import build_model_prompt
        session = self._make_session()
        prior = [{"session_id": "old-1", "disposition": "intake_complete"}]
        prompt = build_model_prompt(session, [], [], [], prior_calls_context=prior)
        self.assertIn("prior_calls_context", prompt)

    def test_no_rag_context_omits_keys(self):
        from voice_intake.prompting import build_model_prompt
        session = self._make_session()
        prompt = build_model_prompt(session, [], [], [])
        self.assertNotIn("knowledge_context", prompt)
        self.assertNotIn("patient_context", prompt)
        self.assertNotIn("eligibility_context", prompt)
        self.assertNotIn("prior_calls_context", prompt)


if __name__ == "__main__":
    unittest.main()
