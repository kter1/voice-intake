from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class EligibilityResult:
    eligible: bool
    plan_name: str
    copay: str | None
    notes: str


@runtime_checkable
class EligibilityInterface(Protocol):
    def check_eligibility(
        self,
        member_id: str,
        dob: str,
        insurance_plan: str,
    ) -> EligibilityResult: ...


class MockEligibilityAdapter:
    """Returns eligible=True for any input. Used when MOCK_ELIGIBILITY=true."""

    def check_eligibility(
        self,
        member_id: str,
        dob: str,
        insurance_plan: str,
    ) -> EligibilityResult:
        return EligibilityResult(
            eligible=True,
            plan_name=insurance_plan or "Mock Plan",
            copay="$20",
            notes="Mock eligibility check - all inputs approved.",
        )


class AvailityAdapter:
    """
    Real eligibility via Availity REST API.
    Requires AVAILITY_CLIENT_ID and AVAILITY_CLIENT_SECRET in environment.
    Swap MockEligibilityAdapter with this adapter in production.
    """

    def __init__(self, client_id: str, client_secret: str) -> None:
        self._client_id = client_id
        self._client_secret = client_secret

    def check_eligibility(
        self,
        member_id: str,
        dob: str,
        insurance_plan: str,
    ) -> EligibilityResult:
        raise NotImplementedError(
            "AvailityAdapter is not implemented for this environment. "
            "Set MOCK_ELIGIBILITY=true for development."
        )
