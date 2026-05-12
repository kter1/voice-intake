from __future__ import annotations

from fastapi import Depends, HTTPException, Request

from voice_intake.config import Settings
from voice_intake.db import SQLAuditStore
from voice_intake.knowledge.retrieval import KnowledgeRetriever
from voice_intake.models import CallSession
from voice_intake.orchestrator import VoiceIntakeOrchestrator
from voice_intake.policy import PolicyEngine, PolicyProfile, demo_policy_profile
from voice_intake.templates import DEFAULT_TEMPLATE_BUNDLE
from voice_intake.validator import ProposalValidator


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_store(request: Request) -> SQLAuditStore:
    return SQLAuditStore(
        request.app.state.session_factory,
        hmac_secret=request.app.state.audit_secret,
    )


def get_orchestrator(
    request: Request,
    store: SQLAuditStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> VoiceIntakeOrchestrator:
    use_demo_policy = settings.policy_profile == "demo" or settings.demo_router
    profile = demo_policy_profile() if use_demo_policy else PolicyProfile()
    policy_engine = PolicyEngine(profile)
    validator = ProposalValidator(DEFAULT_TEMPLATE_BUNDLE, policy_engine)
    return VoiceIntakeOrchestrator(
        policy_profile=profile,
        validator=validator,
        audit_store=store,
        policy_engine=policy_engine,
    )


def get_demo_router(request: Request) -> object | None:
    """Return the app-state DemoRouter singleton, or None if not configured."""
    return getattr(request.app.state, "demo_router", None)


def get_safety_pre_router(request: Request) -> object:
    router = getattr(request.app.state, "safety_pre_router", None)
    if router is None:
        from voice_intake.safety_prerouter import SafetyPreRouter

        router = SafetyPreRouter()
        request.app.state.safety_pre_router = router
    return router


def get_opening_intent_router(request: Request) -> object:
    """Return the OpeningIntentRouter singleton for static meta-question matching."""
    router = getattr(request.app.state, "opening_intent_router", None)
    if router is None:
        from voice_intake.opening_intent_router import OpeningIntentRouter

        router = OpeningIntentRouter()
        request.app.state.opening_intent_router = router
    return router


def load_session_or_404(
    session_id: str,
    store: SQLAuditStore = Depends(get_store),
) -> CallSession:
    session = store.load_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")
    return session


def get_knowledge_retriever(request: Request) -> KnowledgeRetriever | None:
    state = request.app.state
    # Honor MOCK_RAG: skip retriever construction entirely. Without this guard,
    # a None app.state.retriever would cause this dependency to re-instantiate
    # the retriever from chroma_client, defeating the lifespan-level mock.
    settings = getattr(state, "settings", None)
    if settings is not None and getattr(settings, "mock_rag", False):
        return None
    retriever = getattr(state, "retriever", None)
    if retriever is not None:
        return retriever
    chroma_client = getattr(state, "chroma_client", None)
    if chroma_client is None:
        return None
    state.retriever = KnowledgeRetriever(
        chroma_client,
        embed_fn=getattr(state, "embed_fn", None),
    )
    return state.retriever


def get_llm_client(request: Request) -> object | None:
    return getattr(request.app.state, "llm_client", None)
