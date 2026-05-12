"""
Tests for OpeningIntentRouter - static meta-question matching.

Ensures the fast path:
1. Matches exactly 5 hardcoded opening/meta patterns
2. Rejects scheduling, field collection, RAG, and arbitrary input
3. Is state-locked to OPENING only
4. Is profile-locked to demo policy only
5. Is blocked when demo_router=True (preserve DemoRouter exclusivity)
6. Returns normal ModelProposal objects through validator/audit
"""

import pytest

from voice_intake.models import CallState
from voice_intake.opening_intent_router import OpeningIntentRouter


@pytest.fixture
def router():
    return OpeningIntentRouter()


class TestOpeningIntentRouterPatternMatching:
    """Test that the router matches only the 5 hardcoded static patterns."""

    def test_who_are_you_identity_pattern(self, router):
        """'Who are you?' matches identity pattern."""
        proposal = router.match(
            transcript="Who are you?",
            current_state=CallState.OPENING,
            policy_profile_id="demo",
            demo_router_enabled=False,
        )
        assert proposal is not None
        assert proposal.template_id == "opening_disclosure"
        assert proposal.requested_transition == CallState.OPENING
        assert proposal.model_version == "opening-intent-v1"
        assert proposal.variables == {}

    def test_what_are_you_identity_pattern(self, router):
        """'What are you?' matches identity pattern."""
        proposal = router.match(
            transcript="What are you?",
            current_state=CallState.OPENING,
            policy_profile_id="demo",
            demo_router_enabled=False,
        )
        assert proposal is not None
        assert proposal.template_id == "opening_disclosure"

    def test_is_this_an_ai_identity_pattern(self, router):
        """'Is this an AI?' matches identity pattern."""
        proposal = router.match(
            transcript="Is this an AI?",
            current_state=CallState.OPENING,
            policy_profile_id="demo",
            demo_router_enabled=False,
        )
        assert proposal is not None
        assert proposal.template_id == "opening_disclosure"

    def test_what_can_you_do_capabilities_pattern(self, router):
        """'What can you do?' matches capabilities pattern."""
        proposal = router.match(
            transcript="What can you do?",
            current_state=CallState.OPENING,
            policy_profile_id="demo",
            demo_router_enabled=False,
        )
        assert proposal is not None
        assert proposal.template_id == "demo_capabilities_prompt"
        assert proposal.requested_transition == CallState.OPENING

    def test_how_can_you_help_capabilities_pattern(self, router):
        """'How can you help?' matches capabilities pattern."""
        proposal = router.match(
            transcript="How can you help?",
            current_state=CallState.OPENING,
            policy_profile_id="demo",
            demo_router_enabled=False,
        )
        assert proposal is not None
        assert proposal.template_id == "demo_capabilities_prompt"


class TestOpeningIntentRouterCaseInsensitivity:
    """Test that patterns are case-insensitive."""

    def test_uppercase_who_are_you(self, router):
        """Uppercase 'WHO ARE YOU?' matches."""
        proposal = router.match(
            transcript="WHO ARE YOU?",
            current_state=CallState.OPENING,
            policy_profile_id="demo",
            demo_router_enabled=False,
        )
        assert proposal is not None

    def test_mixed_case_what_are_you(self, router):
        """Mixed case 'WhAt ArE yOu?' matches."""
        proposal = router.match(
            transcript="WhAt ArE yOu?",
            current_state=CallState.OPENING,
            policy_profile_id="demo",
            demo_router_enabled=False,
        )
        assert proposal is not None

    def test_lowercase_is_this_an_ai(self, router):
        """Lowercase 'is this an ai?' matches."""
        proposal = router.match(
            transcript="is this an ai?",
            current_state=CallState.OPENING,
            policy_profile_id="demo",
            demo_router_enabled=False,
        )
        assert proposal is not None


class TestOpeningIntentRouterPunctuationTolerance:
    """Test that patterns handle various punctuation."""

    def test_who_are_you_no_punctuation(self, router):
        """'Who are you' without punctuation matches."""
        proposal = router.match(
            transcript="Who are you",
            current_state=CallState.OPENING,
            policy_profile_id="demo",
            demo_router_enabled=False,
        )
        assert proposal is not None

    def test_who_are_you_exclamation(self, router):
        """'Who are you!' with exclamation matches."""
        proposal = router.match(
            transcript="Who are you!",
            current_state=CallState.OPENING,
            policy_profile_id="demo",
            demo_router_enabled=False,
        )
        assert proposal is not None

    def test_who_are_you_multiple_questions(self, router):
        """'Who are you??' with multiple question marks matches."""
        proposal = router.match(
            transcript="Who are you??",
            current_state=CallState.OPENING,
            policy_profile_id="demo",
            demo_router_enabled=False,
        )
        assert proposal is not None


class TestOpeningIntentRouterRejectsNonMatching:
    """Test that the router rejects patterns it should not match."""

    def test_scheduling_intent_rejected(self, router):
        """Scheduling intent is NOT matched."""
        proposal = router.match(
            transcript="I'd like to schedule an appointment",
            current_state=CallState.OPENING,
            policy_profile_id="demo",
            demo_router_enabled=False,
        )
        assert proposal is None

    def test_appointment_request_rejected(self, router):
        """Appointment request is NOT matched."""
        proposal = router.match(
            transcript="I want to make an appointment",
            current_state=CallState.OPENING,
            policy_profile_id="demo",
            demo_router_enabled=False,
        )
        assert proposal is None

    def test_combined_meta_and_scheduling_rejected(self, router):
        """Meta question with scheduling request is NOT matched."""
        proposal = router.match(
            transcript="Who are you and when can I schedule?",
            current_state=CallState.OPENING,
            policy_profile_id="demo",
            demo_router_enabled=False,
        )
        assert proposal is None

    def test_policy_question_rejected(self, router):
        """Policy/RAG question is NOT matched."""
        proposal = router.match(
            transcript="What is your cancellation policy?",
            current_state=CallState.OPENING,
            policy_profile_id="demo",
            demo_router_enabled=False,
        )
        assert proposal is None

    def test_field_collection_name_rejected(self, router):
        """Field collection (name) is NOT matched."""
        proposal = router.match(
            transcript="My name is John",
            current_state=CallState.OPENING,
            policy_profile_id="demo",
            demo_router_enabled=False,
        )
        assert proposal is None

    def test_field_collection_dob_rejected(self, router):
        """Field collection (DOB) is NOT matched."""
        proposal = router.match(
            transcript="My DOB is January 1, 1990",
            current_state=CallState.OPENING,
            policy_profile_id="demo",
            demo_router_enabled=False,
        )
        assert proposal is None

    def test_field_collection_insurance_rejected(self, router):
        """Field collection (insurance) is NOT matched."""
        proposal = router.match(
            transcript="I have Aetna",
            current_state=CallState.OPENING,
            policy_profile_id="demo",
            demo_router_enabled=False,
        )
        assert proposal is None

    def test_arbitrary_question_rejected(self, router):
        """Arbitrary question is NOT matched."""
        proposal = router.match(
            transcript="What's your favorite color?",
            current_state=CallState.OPENING,
            policy_profile_id="demo",
            demo_router_enabled=False,
        )
        assert proposal is None

    def test_general_conversation_rejected(self, router):
        """General conversation is NOT matched."""
        proposal = router.match(
            transcript="Hello, how are you today?",
            current_state=CallState.OPENING,
            policy_profile_id="demo",
            demo_router_enabled=False,
        )
        assert proposal is None


class TestOpeningIntentRouterStateLocking:
    """Test that the router is strictly state-locked to OPENING."""

    def test_from_identity_capture_returns_none(self, router):
        """From IDENTITY_CAPTURE state, returns None."""
        proposal = router.match(
            transcript="Who are you?",
            current_state=CallState.IDENTITY_CAPTURE,
            policy_profile_id="demo",
            demo_router_enabled=False,
        )
        assert proposal is None

    def test_from_demographics_returns_none(self, router):
        """From DEMOGRAPHICS state, returns None."""
        proposal = router.match(
            transcript="Who are you?",
            current_state=CallState.DEMOGRAPHICS,
            policy_profile_id="demo",
            demo_router_enabled=False,
        )
        assert proposal is None

    def test_from_reason_for_visit_returns_none(self, router):
        """From REASON_FOR_VISIT state, returns None."""
        proposal = router.match(
            transcript="Who are you?",
            current_state=CallState.REASON_FOR_VISIT,
            policy_profile_id="demo",
            demo_router_enabled=False,
        )
        assert proposal is None

    def test_from_readback_returns_none(self, router):
        """From READBACK state, returns None."""
        proposal = router.match(
            transcript="Who are you?",
            current_state=CallState.READBACK,
            policy_profile_id="demo",
            demo_router_enabled=False,
        )
        assert proposal is None

    def test_from_close_returns_none(self, router):
        """From CLOSE state, returns None."""
        proposal = router.match(
            transcript="Who are you?",
            current_state=CallState.CLOSE,
            policy_profile_id="demo",
            demo_router_enabled=False,
        )
        assert proposal is None


class TestOpeningIntentRouterProfileLocking:
    """Test that the router is strictly profile-locked to demo policy."""

    def test_default_profile_returns_none(self, router):
        """With default policy profile, returns None."""
        proposal = router.match(
            transcript="Who are you?",
            current_state=CallState.OPENING,
            policy_profile_id="default",
            demo_router_enabled=False,
        )
        assert proposal is None

    def test_custom_profile_returns_none(self, router):
        """With custom policy profile, returns None."""
        proposal = router.match(
            transcript="Who are you?",
            current_state=CallState.OPENING,
            policy_profile_id="custom",
            demo_router_enabled=False,
        )
        assert proposal is None


class TestOpeningIntentRouterDemoRouterLocking:
    """Test that the router does not fire when demo_router is enabled."""

    def test_demo_router_enabled_returns_none(self, router):
        """When demo_router=True, returns None (preserve DemoRouter exclusivity)."""
        proposal = router.match(
            transcript="Who are you?",
            current_state=CallState.OPENING,
            policy_profile_id="demo",
            demo_router_enabled=True,
        )
        assert proposal is None


class TestOpeningIntentRouterProposalStructure:
    """Test that returned proposals are well-formed."""

    def test_proposal_has_correct_fields(self, router):
        """Returned proposal has all required fields."""
        proposal = router.match(
            transcript="Who are you?",
            current_state=CallState.OPENING,
            policy_profile_id="demo",
            demo_router_enabled=False,
        )
        assert proposal is not None
        assert hasattr(proposal, "template_id")
        assert hasattr(proposal, "requested_transition")
        assert hasattr(proposal, "variables")
        assert hasattr(proposal, "model_version")

    def test_proposal_variables_empty(self, router):
        """Returned proposal has no variables."""
        proposal = router.match(
            transcript="What can you do?",
            current_state=CallState.OPENING,
            policy_profile_id="demo",
            demo_router_enabled=False,
        )
        assert proposal is not None
        assert proposal.variables == {}

    def test_proposal_model_version_set(self, router):
        """Returned proposal has model_version='opening-intent-v1'."""
        proposal = router.match(
            transcript="How can you help?",
            current_state=CallState.OPENING,
            policy_profile_id="demo",
            demo_router_enabled=False,
        )
        assert proposal is not None
        assert proposal.model_version == "opening-intent-v1"

    def test_proposal_transition_is_opening(self, router):
        """Returned proposal always requests OPENING state."""
        for transcript in [
            "Who are you?",
            "What are you?",
            "Is this an AI?",
            "What can you do?",
            "How can you help?",
        ]:
            proposal = router.match(
                transcript=transcript,
                current_state=CallState.OPENING,
                policy_profile_id="demo",
                demo_router_enabled=False,
            )
            assert proposal is not None
            assert proposal.requested_transition == CallState.OPENING
