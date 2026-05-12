from __future__ import annotations

from dataclasses import dataclass

from .models import CallState
from .state_machine import can_transition


@dataclass(slots=True)
class PartitionMetric:
    partition: str
    wer: float
    critical_slot_digit_accuracy: float


@dataclass(slots=True)
class AsrCertificationResult:
    passed: bool
    errors: list[str]


@dataclass(slots=True)
class PilotGateDefinition:
    min_calls: int = 500
    max_phi_leaks: int = 0
    required_confirmation_rate: float = 1.0
    required_latency_pass: bool = True
    require_transfer_band: bool = True
    require_incident_drill: bool = True


@dataclass(slots=True)
class PilotObservation:
    completed_calls: int
    phi_leak_count: int
    confirmation_rate: float
    latency_met: bool
    transfer_rate_in_band: bool
    incident_drill_passed: bool


@dataclass(slots=True)
class PilotGateResult:
    passed: bool
    failures: list[str]


def evaluate_asr_partitions(metrics: list[PartitionMetric]) -> AsrCertificationResult:
    errors: list[str] = []
    if not metrics:
        return AsrCertificationResult(False, ["no partition metrics provided"])
    wers = [metric.wer for metric in metrics]
    accuracies = [metric.critical_slot_digit_accuracy for metric in metrics]
    if max(wers) - min(wers) > 5.0:
        errors.append("WER disparity exceeds 5 percentage points")
    if max(accuracies) - min(accuracies) > 5.0:
        errors.append("critical-slot accuracy disparity exceeds 5 percentage points")
    return AsrCertificationResult(not errors, errors)


def evaluate_pilot_gate(
    observation: PilotObservation, definition: PilotGateDefinition | None = None
) -> PilotGateResult:
    definition = definition or PilotGateDefinition()
    failures: list[str] = []
    if observation.completed_calls < definition.min_calls:
        failures.append("minimum supervised call count not met")
    if observation.phi_leak_count > definition.max_phi_leaks:
        failures.append("confirmed PHI leakage defects present")
    if observation.confirmation_rate < definition.required_confirmation_rate:
        failures.append("critical-field confirmation rate below threshold")
    if definition.required_latency_pass and not observation.latency_met:
        failures.append("latency target not met")
    if definition.require_transfer_band and not observation.transfer_rate_in_band:
        failures.append("transfer rate outside pre-approved band")
    if definition.require_incident_drill and not observation.incident_drill_passed:
        failures.append("incident response drill not passed")
    return PilotGateResult(not failures, failures)


def validate_transition_history(history: list[CallState]) -> bool:
    if len(history) < 2:
        return True
    for current_state, next_state in zip(history, history[1:]):
        if not can_transition(current_state, next_state):
            return False
    return True

