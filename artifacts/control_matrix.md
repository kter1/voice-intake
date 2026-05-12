# Control Matrix

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
