from __future__ import annotations

import re
from datetime import datetime, timezone

from .models import FieldCandidate, NormalizationResult, NormalizedSegment, VerificationStatus, VoiceTurn


PHONE_RE = re.compile(r"(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){2}\d{4}")
DIGIT_BLOCK_RE = re.compile(r"\b\d{6,20}\b")
DOB_RE = re.compile(r"\b(?:\d{1,2}[/-]){2}\d{2,4}\b")


def normalize_dob(raw: str) -> str | None:
    """Parse a raw DOB string (M/D/YY or M/D/YYYY) and return ISO YYYY-MM-DD.

    Uses a current-year pivot for 2-digit years: years <= (current year % 100)
    are mapped to 20xx; years above the pivot are mapped to 19xx.

    Validates the parsed date through datetime() so calendar-impossible inputs
    (e.g. "2/31/01", "13/1/01", "2/30/2024") return None.

    Returns None on any parse failure.
    """
    cleaned = raw.strip().replace("-", "/")
    parts = cleaned.split("/")
    if len(parts) != 3:
        return None
    month_s, day_s, year_s = (p.strip() for p in parts)
    try:
        month = int(month_s)
        day = int(day_s)
    except ValueError:
        return None
    if not year_s.isdigit():
        # Guard before the pivot's int() call so non-numeric years
        # (e.g. "1/2/aa") return None cleanly.
        return None
    if len(year_s) == 2:
        pivot = datetime.now(timezone.utc).year % 100
        century = "20" if int(year_s) <= pivot else "19"
        year_s = f"{century}{year_s}"
    try:
        year = int(year_s)
    except ValueError:
        return None
    if not (1900 <= year <= 2100):
        return None
    try:
        parsed = datetime(year, month, day, tzinfo=timezone.utc)
    except ValueError:
        # Calendar-invalid (e.g. Feb 30, month=13, day=32)
        return None
    return parsed.strftime("%Y-%m-%d")


def normalize_turn(turn: VoiceTurn) -> NormalizationResult:
    segments: list[NormalizedSegment] = []
    for match in PHONE_RE.finditer(turn.transcript):
        digits = re.sub(r"\D", "", match.group(0))[-10:]
        segments.append(
            NormalizedSegment(
                field_type="callback_number",
                raw_span=match.group(0),
                normalized_value=digits,
                normalization_confidence=0.98,
            )
        )
    for match in DOB_RE.finditer(turn.transcript):
        # Route through normalize_dob so calendar-impossible dates (e.g.
        # "2/31/01") produce no DOB segment, matching the helper's behavior
        # used by DemoRouter.
        normalized = normalize_dob(match.group(0))
        if normalized is None:
            continue
        segments.append(
            NormalizedSegment(
                field_type="dob",
                raw_span=match.group(0),
                normalized_value=normalized,
                normalization_confidence=0.92,
            )
        )
    for match in DIGIT_BLOCK_RE.finditer(turn.transcript):
        digits = match.group(0)
        if len(digits) >= 8:
            segments.append(
                NormalizedSegment(
                    field_type="member_id",
                    raw_span=digits,
                    normalized_value=digits,
                    normalization_confidence=0.90,
                )
            )
    return NormalizationResult(
        turn_id=turn.turn_id,
        raw_transcript=turn.transcript,
        normalized_segments=segments,
    )


def extract_field_candidates(normalization: NormalizationResult) -> list[FieldCandidate]:
    candidates: list[FieldCandidate] = []
    for segment in normalization.normalized_segments:
        candidates.append(
            FieldCandidate(
                field_name=segment.field_type,
                candidate_value=segment.normalized_value,
                confidence=segment.normalization_confidence,
                verification_status=VerificationStatus.UNVERIFIED,
                source_turn_ids=[normalization.turn_id],
                sensitivity_class="phi",
            )
        )
    return candidates

