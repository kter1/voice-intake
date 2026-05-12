"""
Opening Intent Router - Deterministic intent matching for static system identity questions.

This router handles only narrow, static meta-questions from the OPENING state:
- "Who are you?" / "What are you?" / "Is this an AI?" → opening_disclosure
- "What can you do?" / "How can you help?" → demo_capabilities_prompt

Everything else (scheduling, field collection, RAG questions, arbitrary input) flows through
the normal LLM path.

DO NOT EXTEND THIS BEYOND STATIC OPENING META-QUESTIONS.
If you find yourself adding patterns for IDENTITY_CAPTURE/DEMOGRAPHICS/SCHEDULING,
escalate to architecture review - this becomes DemoRouter 2.0.
"""

import re
from typing import Optional

from voice_intake.models import CallState, ModelProposal


class OpeningIntentRouter:
    """Matches narrow, static opening/meta intent patterns."""

    # Exactly 5 hardcoded patterns for system identity questions.
    # No fuzzy matching, no broad patterns, no appointment/scheduling keywords.
    # Patterns tolerate optional punctuation (?, !, or nothing) at the end.
    OPENING_IDENTITY_PATTERNS = [
        r"^who\s+are\s+you[?!]*$",
        r"^what\s+are\s+you[?!]*$",
        r"^is\s+this\s+an\s+ai[?!]*$",
    ]

    OPENING_CAPABILITIES_PATTERNS = [
        r"^what\s+can\s+you\s+do[?!]*$",
        r"^how\s+can\s+you\s+help[?!]*$",
    ]

    def match(
        self,
        transcript: str,
        current_state: CallState,
        policy_profile_id: str,
        demo_router_enabled: bool,
    ) -> Optional[ModelProposal]:
        """
        Check if transcript matches a static opening meta-question.

        Args:
            transcript: Caller's transcript text
            current_state: Current call state
            policy_profile_id: Policy profile (only matches when "demo")
            demo_router_enabled: If true, skip fast path (preserve DemoRouter exclusivity)

        Returns:
            ModelProposal with template_id and requested_transition, or None.
        """
        # Hard scope guards
        if current_state != CallState.OPENING:
            return None
        if policy_profile_id != "demo":
            return None
        if demo_router_enabled:
            return None

        # Normalize: lowercase, strip leading/trailing whitespace
        norm = transcript.strip().lower()

        # Check identity patterns
        for pattern in self.OPENING_IDENTITY_PATTERNS:
            if re.match(pattern, norm):
                return ModelProposal(
                    proposal_id="",  # Will be overwritten by caller
                    session_id="",  # Will be overwritten by caller
                    source_turn_id="",  # Will be overwritten by caller
                    template_id="opening_disclosure",
                    requested_transition=CallState.OPENING,
                    variables={},
                    model_version="opening-intent-v1",
                )

        # Check capabilities patterns
        for pattern in self.OPENING_CAPABILITIES_PATTERNS:
            if re.match(pattern, norm):
                return ModelProposal(
                    proposal_id="",  # Will be overwritten by caller
                    session_id="",  # Will be overwritten by caller
                    source_turn_id="",  # Will be overwritten by caller
                    template_id="demo_capabilities_prompt",
                    requested_transition=CallState.OPENING,
                    variables={},
                    model_version="opening-intent-v1",
                )

        return None
