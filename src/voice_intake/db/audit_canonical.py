from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, date, datetime
from enum import Enum
from typing import Any

from voice_intake.models import AuditEvent

_CHAIN_FIELDS = [
    "event_id",
    "session_id",
    "actor",
    "event_type",
    "proposal_id",
    "validator_result_id",
    "affected_fields",
    "identity_state",
    "policy_decision_id",
    "template_id",
    "model_version",
    "service_path",
    "before_after_hash",
    "timestamp",
    "details",
]


def canonicalize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        if isinstance(value, datetime) and value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    if isinstance(value, dict):
        return {key: canonicalize(value[key]) for key in sorted(value.keys())}
    raise TypeError(f"Cannot canonicalize {type(value).__name__}")


def canonical_event_json(event: AuditEvent) -> str:
    payload = {field: canonicalize(getattr(event, field, None)) for field in _CHAIN_FIELDS}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def compute_chain_hash(secret: bytes, previous_chain_hash: str, event: AuditEvent) -> str:
    return hmac.new(
        secret,
        (previous_chain_hash + canonical_event_json(event)).encode(),
        hashlib.sha256,
    ).hexdigest()
