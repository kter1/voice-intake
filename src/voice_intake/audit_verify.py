from __future__ import annotations

import hmac

from voice_intake.db.audit_canonical import compute_chain_hash
from voice_intake.models import AuditEvent


def verify_chain(events: list[AuditEvent], secret: str) -> list[str]:
    non_legacy = [event for event in events if event.chain_index is not None and event.chain_hash]
    non_legacy.sort(key=lambda event: event.chain_index or 0)
    previous_hash = ""
    errors: list[str] = []
    for event in non_legacy:
        expected = compute_chain_hash(secret.encode(), previous_hash, event)
        if not hmac.compare_digest(expected, event.chain_hash):
            errors.append(
                f"event_id={event.event_id} chain_hash mismatch at chain_index={event.chain_index}"
            )
        previous_hash = event.chain_hash
    return errors
