"""Deterministic mock insurance network checker for the public scheduling demo.

These sets are intentionally small and illustrative - this is not a real payer
directory.  The demo discloses that network status is mock ("for this demo").
"""
from __future__ import annotations

import re

from .models import InsuranceNetworkStatus

# Canonical lowercase names that are in-network for the demo
IN_NETWORK: frozenset[str] = frozenset(
    {
        "aetna",
        "blue cross",
        "blue shield",
        "bcbs",
        "cigna",
        "united healthcare",
        "uhc",
    }
)

# Canonical lowercase names that are out-of-network for the demo
OUT_OF_NETWORK: frozenset[str] = frozenset({"kaiser", "oscar", "ambetter"})

# Expressions indicating the caller has no insurance / will self-pay.
# Entries must be in POST-normalization form: _normalize() strips punctuation
# (including hyphens), so "self-pay" → "selfpay" before the set lookup.
SELF_PAY: frozenset[str] = frozenset(
    {"none", "self pay", "selfpay", "no insurance", "uninsured"}
)

_STRIP_RE = re.compile(r"[^\w\s]")  # strip punctuation except spaces/word chars
_DISPLAY_SAFE_RE = re.compile(r"[^A-Za-z0-9 \-&'.]")


def _normalize(raw: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    stripped = _STRIP_RE.sub("", raw.lower()).strip()
    return " ".join(stripped.split())


def _safe_unknown_display(raw: str) -> str:
    safe = _DISPLAY_SAFE_RE.sub("", raw.strip())
    safe = " ".join(safe.split())
    return safe.title() if safe else "Unknown"


def check_network_status(raw: str) -> tuple[InsuranceNetworkStatus, str]:
    """Return (network_status, display_name) for a raw insurance string.

    Display name is the raw value title-cased when unknown, or the canonical
    form for known entries.  Self-pay / no-insurance returns NOT_PROVIDED with
    display name "self-pay".
    """
    norm = _normalize(raw)

    if norm in SELF_PAY:
        return InsuranceNetworkStatus.NOT_PROVIDED, "self-pay"

    if norm in IN_NETWORK:
        # Return title-cased canonical form (e.g. "Aetna", "Blue Cross")
        display = " ".join(w.capitalize() for w in norm.split())
        return InsuranceNetworkStatus.IN_NETWORK, display

    if norm in OUT_OF_NETWORK:
        display = " ".join(w.capitalize() for w in norm.split())
        return InsuranceNetworkStatus.OUT_OF_NETWORK, display

    # Unknown: preserve caller wording where possible, but keep template-safe chars.
    display = _safe_unknown_display(raw)
    return InsuranceNetworkStatus.UNKNOWN, display
