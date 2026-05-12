"""Twilio-specific utilities: signature validation, STIR/SHAKEN, TwiML generation."""

from __future__ import annotations

import logging

from voice_intake.models import StirShakenAttestation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# STIR/SHAKEN
# ---------------------------------------------------------------------------

# Twilio StirVerstat field values → our domain attestation
_STIR_MAP: dict[str, StirShakenAttestation] = {
    "TN-Validation-Passed-A": StirShakenAttestation.A,
    "TN-Validation-Passed-B": StirShakenAttestation.B,
    "TN-Validation-Passed-C": StirShakenAttestation.C,
}


def parse_stir_verstat(verstat: str | None) -> StirShakenAttestation:
    """Convert Twilio's StirVerstat value to our domain attestation enum.

    Any unrecognised value (including None) maps to UNKNOWN.
    """
    if not verstat:
        return StirShakenAttestation.UNKNOWN
    return _STIR_MAP.get(verstat, StirShakenAttestation.UNKNOWN)


# ---------------------------------------------------------------------------
# Signature validation
# ---------------------------------------------------------------------------


def validate_twilio_signature(
    auth_token: str,
    signature: str,
    url: str,
    params: dict[str, str],
) -> bool:
    """Validate an inbound Twilio webhook request signature.

    Uses ``twilio.request_validator.RequestValidator`` from the Twilio SDK.
    Returns False (rather than raising) on any error so callers can return 403.
    """
    try:
        from twilio.request_validator import RequestValidator  # type: ignore[import]

        validator = RequestValidator(auth_token)
        return bool(validator.validate(url, params, signature))
    except Exception:
        logger.exception("Twilio signature validation raised an exception")
        return False


# ---------------------------------------------------------------------------
# TwiML builders
# ---------------------------------------------------------------------------


def build_stream_twiml(stream_url: str) -> str:
    """Return TwiML that opens a bidirectional Media Stream to *stream_url*.

    Twilio will connect the call audio to our WebSocket handler and keep the
    call live until the WebSocket closes or the caller hangs up.
    """
    safe_url = stream_url.replace('"', "&quot;")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Connect>"
        f'<Stream url="{safe_url}"/>'
        "</Connect>"
        "</Response>"
    )


def build_say_twiml(text: str, voice: str = "Polly.Joanna") -> str:
    """Return TwiML that speaks *text* back to the caller using Amazon Polly."""
    safe = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f'<Say voice="{voice}">{safe}</Say>'
        "</Response>"
    )


def build_dial_twiml(queue_name: str = "human-agent-queue") -> str:
    """Return TwiML that transfers the call to a human agent queue.

    Called when the orchestrator reaches HUMAN_TAKEOVER state.
    """
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"<Dial><Queue>{queue_name}</Queue></Dial>"
        "</Response>"
    )
