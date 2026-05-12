from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import FieldCandidate


@dataclass(frozen=True, slots=True)
class PromptBoundaryPolicy:
    allowed: dict[str, list[str]]
    forbidden_patterns: tuple[re.Pattern[str], ...]
    forbidden_field_types: frozenset[str]
    source_hash: str
    source_path: Path


class PromptBoundaryError(ValueError):
    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__("; ".join(violations))


def _parse_policy_yaml(raw_text: str) -> dict[str, dict[str, list[str]]]:
    parsed: dict[str, dict[str, list[str]]] = {}
    section: str | None = None
    key: str | None = None
    for raw_line in raw_text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            if not line.endswith(":"):
                raise ValueError("prompt boundary policy must use section mappings")
            section = line[:-1]
            parsed[section] = {}
            key = None
            continue
        if section is None:
            raise ValueError("prompt boundary policy item appears before a section")
        stripped = line.strip()
        if stripped.endswith(":"):
            key = stripped[:-1]
            parsed[section][key] = []
            continue
        if stripped.startswith("- "):
            if key is None:
                raise ValueError("prompt boundary policy list item appears before a key")
            value = stripped[2:].strip()
            if (
                len(value) >= 2
                and value[0] == value[-1]
                and value[0] in {"'", '"'}
            ):
                value = value[1:-1]
            parsed[section][key].append(value)
            continue
        raise ValueError("prompt boundary policy contains unsupported YAML")
    return parsed


def load_prompt_boundary_policy(path: Path) -> PromptBoundaryPolicy:
    raw_text = path.read_text(encoding="utf-8")
    parsed = _parse_policy_yaml(raw_text)
    allowed = parsed.get("allowed", {})
    forbidden = parsed.get("forbidden", {})
    patterns = forbidden.get("patterns", [])
    field_types = forbidden.get("field_types", [])
    return PromptBoundaryPolicy(
        allowed={key: list(value) for key, value in allowed.items()},
        forbidden_patterns=tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns),
        forbidden_field_types=frozenset(field_types),
        source_hash=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        source_path=path,
    )


def _placeholder(field_name: str) -> str:
    return f"[{field_name.upper()}]"


def sanitize_text(
    text: str,
    field_candidates: list[FieldCandidate],
    policy: PromptBoundaryPolicy,
) -> str:
    sanitized = text
    for candidate in field_candidates:
        if candidate.field_name in policy.forbidden_field_types and candidate.candidate_value:
            sanitized = re.sub(
                re.escape(candidate.candidate_value),
                _placeholder(candidate.field_name),
                sanitized,
                flags=re.IGNORECASE,
            )
    for pattern in policy.forbidden_patterns:
        sanitized = pattern.sub("[REDACTED]", sanitized)
    return sanitized


def sanitize_prompt_payload(
    prompt: dict[str, Any],
    field_candidates: list[FieldCandidate],
    policy: PromptBoundaryPolicy,
) -> dict[str, Any]:
    sanitized = copy.deepcopy(prompt)
    recent_turns = sanitized.get("recent_turns")
    if isinstance(recent_turns, list):
        for turn in recent_turns:
            if isinstance(turn, dict) and isinstance(turn.get("transcript"), str):
                turn["transcript"] = sanitize_text(turn["transcript"], field_candidates, policy)
    return sanitized


def inspect_prompt_payload(
    prompt: dict[str, Any],
    field_candidates: list[FieldCandidate],
    policy: PromptBoundaryPolicy,
) -> None:
    violations: list[str] = []
    serialized = json.dumps(prompt, sort_keys=True)
    for candidate in field_candidates:
        if (
            candidate.field_name in policy.forbidden_field_types
            and candidate.candidate_value
            and candidate.candidate_value in serialized
        ):
            violations.append(f"unmasked {candidate.field_name} present in prompt payload")
    for pattern in policy.forbidden_patterns:
        if pattern.search(serialized):
            violations.append(f"forbidden pattern leaked into prompt payload: {pattern.pattern}")
    if violations:
        raise PromptBoundaryError(violations)
