from __future__ import annotations

import json
from pathlib import Path

from .evaluation import PilotGateDefinition
from .state_machine import ALLOWED_TRANSITIONS


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"


def export_state_machine() -> None:
    payload = {
        state.value: sorted(target.value for target in targets)
        for state, targets in ALLOWED_TRANSITIONS.items()
    }
    (ARTIFACTS / "state_machine.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def export_control_matrix() -> None:
    markdown = """# Control Matrix

| Subsystem | Core controls |
| --- | --- |
| Telephony adapter | STIR/SHAKEN capture, call metadata retention policy, audio reference control |
| Normalization and extraction | Deterministic normalization, field-candidate provenance, no policy bypass |
| Dialog model | Bounded prompt context, masked confirmation values only, no raw field values |
| Validator | Reject-only enforcement, closed reject reasons, deterministic state transition checks |
| Policy engine | Pinned profile, consent-aware behavior, fraud posture derived from STIR/SHAKEN |
| Template bundle | Compliance-reviewed publication, version pinning per session, behavioral suite gate |
| Orchestrator | Hold-template timeout handling, buffered caller audio preservation, takeover forcing |
| Audit store | Persisted proposals, validator results, policy decisions, consent artifacts, interventions |
| Supervisor console | Explicit write surface, audited overrides, active alerting in manual and emergency states |
| Operations | MFA, encryption, segmentation, vuln scans, pen tests, patch tiers, restore drill |
"""
    (ARTIFACTS / "control_matrix.md").write_text(markdown, encoding="utf-8")


def export_evaluation_harness() -> None:
    gate = PilotGateDefinition()
    markdown = f"""# Evaluation Harness

## Deterministic Suites

- Template-lock enforcement
- PHI-before-verification leakage
- State-transition legality
- Validator reject-reason coverage
- Audit-chain completeness
- Prompt-injection blocking at validator and policy boundary

## Pilot Gates

- Minimum supervised calls: {gate.min_calls}
- Maximum confirmed PHI leaks: {gate.max_phi_leaks}
- Required critical-field confirmation rate: {gate.required_confirmation_rate:.0%}
- Real-call latency target: required
- Transfer rate: must remain inside pre-approved band
- Incident response and restoration drill: required
"""
    (ARTIFACTS / "evaluation_harness.md").write_text(markdown, encoding="utf-8")


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    export_state_machine()
    export_control_matrix()
    export_evaluation_harness()


if __name__ == "__main__":
    main()
