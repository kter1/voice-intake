"""Deterministic reject-only guard for model-authored spoken text (C4.6b).

The ProposalValidator still gates template_id, variables, and state
transitions - nothing here weakens that. This guard decides only whether the
model's optional natural-language phrasing (``spoken_response``) may be spoken
INSTEAD of the approved template content. Any failure silently falls back to
the template text; it never rejects the proposal itself.

This is demo-grade screening for the synthetic portfolio demo (length bounds,
unmasked digit-run detection, markup rejection, clinical-language deny list).
It does NOT detect names, addresses, or other non-numeric PHI, and the deny
list is a small keyword set, not clinical NLP. A production healthcare
deployment would require a reviewed guard policy.
"""

from __future__ import annotations

import re

MAX_SPOKEN_CHARS = 320

_DIGIT_RUN = re.compile(r"\d{7,}")
_SSN_LIKE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# Angle brackets and braces are rejected outright: spoken text must be plain
# prose, never markup, TwiML, or unresolved {{placeholders}}.
_MARKUP = re.compile(r"[<>{}]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

_CLINICAL_DENYLIST = (
    "diagnos",
    "prescrib",
    "dosage",
    "dose of",
    "medical advice",
    "you should take",
    "stop taking",
)


def guard_spoken_text(spoken_text: str | None) -> tuple[str | None, str]:
    """Screen model-authored spoken text.

    Returns (approved_text, reason). approved_text is None whenever the text
    is absent or rejected; reason is "accepted", "absent", or the rejection
    cause (for the audit trail).
    """
    if not spoken_text or not spoken_text.strip():
        return None, "absent"
    text = " ".join(spoken_text.split())
    if len(text) > MAX_SPOKEN_CHARS:
        return None, "too_long"
    if _CONTROL.search(text) or _MARKUP.search(text):
        return None, "markup"
    if _DIGIT_RUN.search(text) or _SSN_LIKE.search(text):
        return None, "unmasked_digits"
    lowered = text.lower()
    for term in _CLINICAL_DENYLIST:
        if term in lowered:
            return None, "clinical_language"
    return text, "accepted"
