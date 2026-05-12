from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from .models import CallSession, ModelProposal, PolicyDecision, RejectReason, ValidatorResult
from .policy import PolicyEngine
from .state_machine import can_transition
from .templates import Template, TemplateBundle


def _new_id() -> str:
    return str(uuid4())


@dataclass(slots=True)
class ValidationOutcome:
    validator_result: ValidatorResult
    policy_decision: PolicyDecision | None
    template: Template | None


class ProposalValidator:
    def __init__(self, template_bundle: TemplateBundle, policy_engine: PolicyEngine) -> None:
        self.template_bundle = template_bundle
        self.policy_engine = policy_engine

    def validate(self, session: CallSession, proposal: ModelProposal) -> ValidationOutcome:
        template = self.template_bundle.template_for(proposal.template_id)
        if template is None or session.current_state not in template.allowed_states:
            return self._reject(proposal.proposal_id, RejectReason.INVALID_TEMPLATE_ID)
        if not can_transition(session.current_state, proposal.requested_transition):
            return self._reject(proposal.proposal_id, RejectReason.DISALLOWED_TRANSITION)

        allowed_names = set(template.allowed_variables)
        provided_names = set(proposal.variables)
        if provided_names != allowed_names:
            return self._reject(proposal.proposal_id, RejectReason.VARIABLE_SCHEMA_VIOLATION)
        for name, value in proposal.variables.items():
            spec = template.allowed_variables[name]
            if not spec.validate(value):
                reject_reason = (
                    RejectReason.OUT_OF_VOCABULARY_PHI
                    if spec.masked_only
                    else RejectReason.VARIABLE_SCHEMA_VIOLATION
                )
                return self._reject(proposal.proposal_id, reject_reason)

        decision = self.policy_engine.decide(
            session=session,
            template_id=proposal.template_id,
            requested_action=proposal.requested_transition.value,
        )
        if not decision.allowed:
            return ValidationOutcome(
                validator_result=ValidatorResult(
                    validator_result_id=_new_id(),
                    proposal_id=proposal.proposal_id,
                    accepted=False,
                    reject_reason=RejectReason.POLICY_DENIED,
                ),
                policy_decision=decision,
                template=template,
            )
        return ValidationOutcome(
            validator_result=ValidatorResult(
                validator_result_id=_new_id(),
                proposal_id=proposal.proposal_id,
                accepted=True,
                reject_reason=None,
            ),
            policy_decision=decision,
            template=template,
        )

    def _reject(self, proposal_id: str, reason: RejectReason) -> ValidationOutcome:
        return ValidationOutcome(
            validator_result=ValidatorResult(
                validator_result_id=_new_id(),
                proposal_id=proposal_id,
                accepted=False,
                reject_reason=reason,
            ),
            policy_decision=None,
            template=None,
        )

