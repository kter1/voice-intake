"""
Unit tests for voice_intake.normalization - DOB pivot + invalid calendar
date rejection. Phase 4.5 regression coverage.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from voice_intake.models import Speaker, VoiceTurn
from voice_intake.normalization import normalize_dob, normalize_turn


# ── normalize_dob: pivot behavior ─────────────────────────────────────────────


class _FakeDatetime:
    """Stand-in for datetime that lets tests stub `datetime.now()` for the pivot
    while leaving construction (`datetime(year, month, day)`) untouched."""

    def __init__(self, fixed_year: int) -> None:
        self._fixed_year = fixed_year

    def now(self, tz=None):  # noqa: D401 - match signature
        return datetime(self._fixed_year, 6, 15, tzinfo=tz or timezone.utc)


def _stub_now(monkeypatch, year: int) -> None:
    """Patch normalization.datetime.now() to return a date in the given year."""
    fake = _FakeDatetime(year)
    # Replace the `datetime` symbol that normalize_dob imports at module scope.
    import voice_intake.normalization as norm

    class _Wrap(datetime):
        @classmethod
        def now(cls, tz=None):
            return fake.now(tz)

    monkeypatch.setattr(norm, "datetime", _Wrap)


def test_normalize_dob_pivot_current_year_yields_2000s_century(monkeypatch):
    _stub_now(monkeypatch, 2026)
    assert normalize_dob("8/7/01") == "2001-08-07"


def test_normalize_dob_pivot_above_year_yields_1900s_century(monkeypatch):
    _stub_now(monkeypatch, 2026)
    assert normalize_dob("8/7/27") == "1927-08-07"


def test_normalize_dob_pivot_at_boundary(monkeypatch):
    _stub_now(monkeypatch, 2026)
    # Year == pivot (26) → 2000s
    assert normalize_dob("8/7/26") == "2026-08-07"


def test_normalize_dob_4_digit_year_unchanged(monkeypatch):
    _stub_now(monkeypatch, 2026)
    assert normalize_dob("8/7/1985") == "1985-08-07"


def test_normalize_dob_pivot_rolls_forward_in_2027(monkeypatch):
    _stub_now(monkeypatch, 2027)
    # 26 <= 27 → 2026 (still 2000s); 28 > 27 → 1928
    assert normalize_dob("8/7/26") == "2026-08-07"
    assert normalize_dob("8/7/28") == "1928-08-07"


# ── normalize_dob: invalid inputs ─────────────────────────────────────────────


def test_normalize_dob_invalid_calendar_dates_return_none(monkeypatch):
    _stub_now(monkeypatch, 2026)
    assert normalize_dob("2/31/01") is None
    assert normalize_dob("13/1/01") is None
    assert normalize_dob("2/30/2024") is None


def test_normalize_dob_garbage_returns_none(monkeypatch):
    _stub_now(monkeypatch, 2026)
    assert normalize_dob("") is None
    assert normalize_dob("abc") is None
    assert normalize_dob("1/2") is None
    assert normalize_dob("1/2/3/4") is None


def test_normalize_dob_non_numeric_year_returns_none(monkeypatch):
    """Regression for the isdigit() guard added in §1.4 Part A."""
    _stub_now(monkeypatch, 2026)
    assert normalize_dob("1/2/aa") is None
    assert normalize_dob("1/2/abc") is None
    assert normalize_dob("1/2/2x") is None


# ── normalize_turn DOB consistency ───────────────────────────────────────────


def _make_turn(transcript: str) -> VoiceTurn:
    now = datetime.now(timezone.utc)
    return VoiceTurn(
        turn_id="t-test",
        speaker=Speaker.CALLER,
        audio_ref=None,
        retention_policy_id="standard",
        transcript=transcript,
        partial_or_final="final",
        asr_confidence=0.9,
        barge_in=False,
        start_ts=now,
        end_ts=now,
    )


def test_normalize_turn_skips_invalid_dob_segments(monkeypatch):
    """Regression for §1.4 Part B: invalid calendar dates produce no DOB segment."""
    _stub_now(monkeypatch, 2026)
    turn = _make_turn("my dob is 2/31/01")
    result = normalize_turn(turn)
    dob_segments = [s for s in result.normalized_segments if s.field_type == "dob"]
    assert dob_segments == []


def test_normalize_turn_keeps_valid_dob_segments(monkeypatch):
    _stub_now(monkeypatch, 2026)
    turn = _make_turn("dob 8/7/01")
    result = normalize_turn(turn)
    dob_segments = [s for s in result.normalized_segments if s.field_type == "dob"]
    assert len(dob_segments) == 1
    assert dob_segments[0].normalized_value == "2001-08-07"
