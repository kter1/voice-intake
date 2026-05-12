from __future__ import annotations

import hashlib
import hmac
import logging
import time

from voice_intake.config import Settings

_log = logging.getLogger(__name__)

DEV_AUDIT_SECRET = "dev-audit-secret-do-not-use-in-production"
DEV_STREAM_SECRET = "dev-stream-secret-do-not-use-in-production"


def verify_api_key(provided: str | None, expected: str) -> bool:
    if not expected:
        return True
    if not provided:
        return False
    return hmac.compare_digest(provided, expected)


def make_stream_token(session_id: str, secret: str, ttl: int = 300) -> str:
    exp = int(time.time()) + ttl
    payload = f"{session_id}:{exp}"
    sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}:{sig}"


def verify_stream_token(token: str, session_id: str, secret: str) -> bool:
    try:
        sid, exp_str, sig = token.rsplit(":", 2)
        if sid != session_id or int(exp_str) < int(time.time()):
            return False
        expected = hmac.new(
            secret.encode(),
            f"{sid}:{exp_str}".encode(),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(sig, expected)
    except Exception:
        return False


def resolve_runtime_secrets(settings: Settings) -> tuple[str, str]:
    if settings.api_key:
        if not settings.audit_hmac_secret:
            raise RuntimeError("AUDIT_HMAC_SECRET is required when API_KEY is set")
        twilio_real = (
            not settings.mock_twilio_validate
            and settings.twilio_account_sid
            and settings.twilio_auth_token
        )
        if twilio_real and not settings.stream_auth_secret:
            raise RuntimeError("STREAM_AUTH_SECRET is required for real Twilio streaming")
        return (
            settings.audit_hmac_secret,
            settings.stream_auth_secret or DEV_STREAM_SECRET,
        )

    if not settings.audit_hmac_secret:
        _log.warning("AUDIT_HMAC_SECRET is blank - using dev-only fallback (NOT for production)")
    if not settings.stream_auth_secret:
        _log.warning("STREAM_AUTH_SECRET is blank - using dev-only fallback (NOT for production)")
    return (
        settings.audit_hmac_secret or DEV_AUDIT_SECRET,
        settings.stream_auth_secret or DEV_STREAM_SECRET,
    )
