"""
Unit tests for voice_intake.insurance.check_network_status - the deterministic
mock network-status checker used by DemoRouter.
"""
from __future__ import annotations

import pytest

from voice_intake.insurance import check_network_status
from voice_intake.models import InsuranceNetworkStatus


@pytest.mark.parametrize("raw, expected_display", [
    ("Aetna", "Aetna"),
    ("aetna", "Aetna"),
    ("BCBS", "Bcbs"),
    ("blue cross", "Blue Cross"),
    ("blue shield", "Blue Shield"),
    ("Cigna", "Cigna"),
    ("united healthcare", "United Healthcare"),
    ("UHC", "Uhc"),
])
def test_check_network_status_in_network(raw, expected_display):
    status, display = check_network_status(raw)
    assert status == InsuranceNetworkStatus.IN_NETWORK
    assert display == expected_display


@pytest.mark.parametrize("raw, expected_display", [
    ("Kaiser", "Kaiser"),
    ("kaiser", "Kaiser"),
    ("Oscar", "Oscar"),
    ("Ambetter", "Ambetter"),
])
def test_check_network_status_out_of_network(raw, expected_display):
    status, display = check_network_status(raw)
    assert status == InsuranceNetworkStatus.OUT_OF_NETWORK
    assert display == expected_display


def test_check_network_status_unknown():
    status, display = check_network_status("FooCare")
    assert status == InsuranceNetworkStatus.UNKNOWN
    # Display preserves user wording (title-cased).
    assert display == "Foocare"


def test_check_network_status_unknown_display_is_template_safe():
    status, display = check_network_status("Aetna PPO with $500 deductible")
    assert status == InsuranceNetworkStatus.UNKNOWN
    assert display == "Aetna Ppo With 500 Deductible"


@pytest.mark.parametrize("raw", [
    "none",
    "None",
    "self pay",
    "self-pay",
    "no insurance",
    "uninsured",
])
def test_check_network_status_self_pay_variants(raw):
    status, display = check_network_status(raw)
    assert status == InsuranceNetworkStatus.NOT_PROVIDED
    assert display == "self-pay"


def test_check_network_status_case_and_punct_normalisation():
    # Punctuation stripped, whitespace collapsed
    status, _ = check_network_status("AETNA!")
    assert status == InsuranceNetworkStatus.IN_NETWORK

    status, _ = check_network_status("  aetna  ")
    assert status == InsuranceNetworkStatus.IN_NETWORK

    # Trailing question mark - the canonical "Aetna?" guardrail. Must still
    # resolve to IN_NETWORK (not UNKNOWN) so DemoRouter's off-script-question
    # detector at INSURANCE doesn't conflict with uncertain answers.
    status, _ = check_network_status("Aetna?")
    assert status == InsuranceNetworkStatus.IN_NETWORK
