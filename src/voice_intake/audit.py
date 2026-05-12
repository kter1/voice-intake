from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .models import AuditEvent, ConsentArtifact, ModelProposal, SupervisorIntervention, ValidatorResult


@runtime_checkable
class AuditStore(Protocol):
    def record_proposal(self, proposal: ModelProposal) -> None: ...
    def record_validator_result(self, result: ValidatorResult) -> None: ...
    def record_policy_decision(self, decision: object) -> None: ...
    def record_audit_event(self, event: AuditEvent) -> None: ...
    def record_consent_artifact(self, artifact: ConsentArtifact) -> None: ...
    def record_supervisor_intervention(self, intervention: SupervisorIntervention) -> None: ...
    def consents_for_session(self, session_id: str) -> list[ConsentArtifact]: ...


@dataclass(slots=True)
class InMemoryAuditStore:
    proposals: list[ModelProposal] = field(default_factory=list)
    validator_results: list[ValidatorResult] = field(default_factory=list)
    policy_decisions: list[object] = field(default_factory=list)
    audit_events: list[AuditEvent] = field(default_factory=list)
    consent_artifacts: list[ConsentArtifact] = field(default_factory=list)
    supervisor_interventions: list[SupervisorIntervention] = field(default_factory=list)

    def record_proposal(self, proposal: ModelProposal) -> None:
        self.proposals.append(proposal)

    def record_validator_result(self, result: ValidatorResult) -> None:
        self.validator_results.append(result)

    def record_policy_decision(self, decision: object) -> None:
        self.policy_decisions.append(decision)

    def record_audit_event(self, event: AuditEvent) -> None:
        self.audit_events.append(event)

    def record_consent_artifact(self, artifact: ConsentArtifact) -> None:
        self.consent_artifacts.append(artifact)

    def record_supervisor_intervention(self, intervention: SupervisorIntervention) -> None:
        self.supervisor_interventions.append(intervention)

    def consents_for_session(self, session_id: str) -> list[ConsentArtifact]:
        return [artifact for artifact in self.consent_artifacts if artifact.session_id == session_id]

