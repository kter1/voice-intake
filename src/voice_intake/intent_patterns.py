from __future__ import annotations

import re

from voice_intake.llm.common import _INJECTION_PATTERNS


def compile_phrases(*phrases: str) -> re.Pattern[str]:
    return re.compile("|".join(phrases), re.IGNORECASE)


EMERGENCY_PHRASES = [
    r"chest\s+pain",
    r"can'?t\s+breathe",
    r"cannot\s+breathe",
    r"trouble\s+breathing",
    r"\bstroke\b",
    r"severe\s+bleeding",
    r"\bbleeding\b",
    r"\bheart\s+attack\b",
    r"\boverdose\b",
    r"\bsuicide\b",
    r"self[\s\-]harm",
    r"going\s+to\s+hurt\s+myself",
    r"\bunconsious\b",
    r"\bunconscious\b",
]

HUMAN_PHRASES = [
    r"\bhuman\b",
    r"\bperson\b",
    r"\brepresentative\b",
    r"\bstaff\b",
    r"\bnurse\b",
    r"\boperator\b",
    r"real\s+person",
    r"talk\s+to\s+someone",
    r"speak\s+to\s+someone",
]

OUT_OF_SCOPE_PHRASES = [
    r"\bbalance\b",
    r"\bbill(?:ing)?\b",
    r"\bpayment\b",
    r"pay\s+(?:my\s+)?(?:bill|balance|invoice)",
    r"how\s+much\s+(?:do\s+)?i\s+owe",
    r"\bowe\b",
    r"\binvoice\b",
    r"\bstatement\b",
    r"\brefund\b",
    r"\bcharge\b",
    r"\bcopay\b",
    r"\bdeductible\b",
    r"\bclaim(?:\s+status)?\b",
]

RE_EMERGENCY = compile_phrases(*EMERGENCY_PHRASES)
RE_HUMAN = compile_phrases(*HUMAN_PHRASES)
RE_OUT_OF_SCOPE = compile_phrases(*OUT_OF_SCOPE_PHRASES)
RE_INJECTION_PATTERNS = [re.compile(pattern) for pattern in _INJECTION_PATTERNS]


def has_prompt_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in RE_INJECTION_PATTERNS)
