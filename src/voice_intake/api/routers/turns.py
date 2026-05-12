from __future__ import annotations

import logging
import time
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, Request

from voice_intake.api.deps import (
    get_demo_router,
    get_knowledge_retriever,
    get_llm_client,
    get_opening_intent_router,
    get_orchestrator,
    get_safety_pre_router,
    get_settings,
    get_store,
    load_session_or_404,
)
from voice_intake.api.schemas.turn import (
    RejectionSchema,
    TurnRequest,
    TurnResponse,
    VoiceActionSchema,
)
from voice_intake.config import Settings
from voice_intake.db import SQLAuditStore
from voice_intake.knowledge.retrieval import KnowledgeRetriever
from voice_intake.llm.dispatch import propose_next_action
from voice_intake.models import CallSession, CallState, Speaker, VoiceTurn, utc_now
from voice_intake.normalization import extract_field_candidates, normalize_turn
from voice_intake.orchestrator import VoiceIntakeOrchestrator
from voice_intake.policy import model_allowed_transitions
from voice_intake.prompting import build_model_prompt

router = APIRouter(tags=["turns"])
logger = logging.getLogger(__name__)


def _new_id() -> str:
    return str(uuid4())


def _build_turn(body: TurnRequest, turn_id: str) -> VoiceTurn:
    ts = utc_now()
    return VoiceTurn(
        turn_id=turn_id,
        speaker=Speaker.CALLER,
        audio_ref=None,
        retention_policy_id="standard",
        transcript=body.transcript,
        partial_or_final=body.partial_or_final,
        asr_confidence=body.asr_confidence,
        barge_in=False,
        start_ts=ts,
        end_ts=ts,
    )


@router.post("/session/{session_id}/turn", response_model=TurnResponse)
async def submit_turn(
    session_id: str,
    body: TurnRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    store: SQLAuditStore = Depends(get_store),
    orchestrator: VoiceIntakeOrchestrator = Depends(get_orchestrator),
    session: CallSession = Depends(load_session_or_404),
    settings: Settings = Depends(get_settings),
    retriever: KnowledgeRetriever | None = Depends(get_knowledge_retriever),
    llm_client: object | None = Depends(get_llm_client),
    demo_router: object | None = Depends(get_demo_router),
    safety_pre_router: object = Depends(get_safety_pre_router),
    opening_intent_router: object = Depends(get_opening_intent_router),
) -> TurnResponse:
    # Latency instrumentation: track each phase of turn processing
    turn_start = time.perf_counter()
    safety_start = None
    safety_end = None
    rag_start = None
    rag_end = None
    prompt_start = None
    prompt_end = None
    llm_start = None
    llm_end = None
    validation_start = None
    validation_end = None
    persistence_start = None
    persistence_end = None

    # Rehydrate double-reject counter from DB into this fresh orchestrator instance
    orchestrator._reject_counts = store.load_reject_counts(session_id)

    # ASR: if audio_b64 is provided and mock_asr=False, decode and transcribe first.
    # An explicit transcript field always takes precedence (pre-transcribed bridges).
    if body.audio_b64 and not settings.mock_asr and not body.transcript:
        import base64
        from voice_intake.asr import get_asr_adapter

        asr = get_asr_adapter(settings)
        audio_bytes = base64.b64decode(body.audio_b64)
        asr_result = await asr.transcribe(audio_bytes)
        # Patch body with the transcribed text and confidence
        body = body.model_copy(update={
            "transcript": asr_result.transcript,
            "asr_confidence": asr_result.confidence,
        })

    turn_id = _new_id()
    turn = _build_turn(body, turn_id)
    store.record_turn(session_id, turn)

    norm = normalize_turn(turn)
    candidates = extract_field_candidates(norm)
    # Filter templates to only those (a) allowed from the current state and
    # (b) selectable by the model.  hold_message and other system/timeout
    # fallback templates set model_facing=False so the LLM never picks them.
    available_templates = [
        template
        for template in orchestrator.validator.template_bundle.templates.values()
        if template.model_facing and session.current_state in template.allowed_states
    ]
    allowed_transitions = model_allowed_transitions(
        orchestrator.policy_profile, session.current_state
    )

    # Generate proposal_id once, used by all routing paths (safety, opening, LLM)
    proposal_id = _new_id()

    # SAFETY FIRST: SafetyPreRouter always wins. Must run before any other routing.
    safety_start = time.perf_counter()
    proposal = None
    if safety_pre_router is not None:
        proposal = safety_pre_router.check(
            transcript=body.transcript,
            current_state=session.current_state,
            source_turn_id=turn_id,
            proposal_id=proposal_id,
            session_id=session_id,
        )
    safety_end = time.perf_counter()

    # Try opening intent router ONLY if safety didn't match
    # (static meta questions from OPENING state, demo policy only)
    if proposal is None and opening_intent_router is not None:
        matched_proposal = opening_intent_router.match(
            transcript=body.transcript,
            current_state=session.current_state,
            policy_profile_id=orchestrator.policy_profile.policy_profile_id,
            demo_router_enabled=settings.demo_router,
        )
        if matched_proposal is not None:
            # Populate proposal identifiers and flow through normal validator/orchestrator
            matched_proposal.proposal_id = proposal_id
            matched_proposal.session_id = session_id
            matched_proposal.source_turn_id = turn_id
            proposal = matched_proposal
            # Fast path: no LLM, no RAG, no prompt build
            llm_start = safety_end
            llm_end = safety_end
            rag_start = safety_end
            rag_end = safety_end
            prompt_start = safety_end
            prompt_end = safety_end

    # If neither safety nor opening fast path matched, proceed with normal LLM flow
    if proposal is None:
        # Build RAG context if retriever is available
        knowledge_chunks = None
        rag_start = time.perf_counter()
        if retriever is not None:
            query = f"{session.current_state.value} {body.transcript[:200]}"
            chunks = retriever.retrieve(query, top_k=3)
            if chunks:
                knowledge_chunks = [
                    {"text": c.text, "source": c.source, "category": c.category, "score": c.score}
                    for c in chunks
                ]
        rag_end = time.perf_counter()

        prompt_start = time.perf_counter()
        prompt = build_model_prompt(
            session,
            [turn],
            candidates,
            available_templates,
            knowledge_chunks=knowledge_chunks,
            allowed_transitions=allowed_transitions,
        )
        prompt_end = time.perf_counter()

        llm_start = time.perf_counter()
        # Pass safety_pre_router=None to prevent double-running safety
        # (already ran at route level above)
        proposal = await propose_next_action(
            prompt,
            source_turn_id=turn_id,
            settings=settings,
            llm_client=llm_client,
            demo_router=demo_router,
            safety_pre_router=None,
            proposal_id=proposal_id,
        )
        llm_end = time.perf_counter()

    if proposal is None:
        # LLM timeout escalates to HUMAN_TAKEOVER. handle_model_timeout() sets
        # current_state=HUMAN_TAKEOVER, mode=HUMAN_TAKEOVER, and records a chained
        # MODEL_TIMEOUT audit event.
        validation_start = validation_start or time.perf_counter()
        validation_end = validation_end or time.perf_counter()
        orchestrator.handle_model_timeout(session, buffered_audio_ref=None)
        persistence_start = persistence_start or time.perf_counter()
        store.save_session(session, reject_counts=orchestrator._reject_counts)
        persistence_end = time.perf_counter()

        # Log latency breakdown
        safety_ms = (safety_end - safety_start) * 1000 if (safety_start and safety_end) else 0
        rag_ms = (rag_end - rag_start) * 1000 if (rag_end and rag_start) else 0
        prompt_ms = (prompt_end - prompt_start) * 1000 if (prompt_end and prompt_start) else 0
        llm_ms = (llm_end - llm_start) * 1000 if (llm_end and llm_start) else 0
        validation_ms = (validation_end - validation_start) * 1000 if (validation_end and validation_start) else 0
        persistence_ms = (persistence_end - persistence_start) * 1000 if (persistence_end and persistence_start) else 0
        total_ms = (time.perf_counter() - turn_start) * 1000
        model_version = getattr(proposal, "model_version", "unknown") if proposal else "unknown"

        logger.info(
            f"turn_latency session_id={session_id} turn_id={turn_id} state={session.current_state.value} "
            f"safety_ms={safety_ms:.1f} rag_ms={rag_ms:.1f} prompt_build_ms={prompt_ms:.1f} "
            f"llm_ms={llm_ms:.1f} validation_ms={validation_ms:.1f} persistence_ms={persistence_ms:.1f} "
            f"total_turn_ms={total_ms:.1f} model_version={model_version} proposal_status=timeout"
        )

        ws_manager = getattr(request.app.state, "supervisor_ws_manager", None)
        if ws_manager is not None:
            background_tasks.add_task(
                ws_manager.broadcast,
                {
                    "type": "takeover_alert",
                    "session_id": session_id,
                    "reason": "llm_timeout",
                    "timestamp": utc_now().isoformat(),
                },
            )
        return TurnResponse(
            session_id=session_id,
            turn_id=turn_id,
            current_state=session.current_state.value,  # now "human_takeover"
            voice_action=None,
            rejection=RejectionSchema(reason="llm_timeout"),
        )

    validation_start = time.perf_counter()
    outcome, action = orchestrator.handle_model_proposal(session, proposal)
    validation_end = time.perf_counter()

    # DemoRouter: commit or discard the pending appointment patch based on
    # validator decision.  Must happen AFTER handle_model_proposal so we
    # know whether the proposal was accepted.  Kept in the API layer (not
    # the orchestrator) to avoid demo-specific concerns leaking into core logic.
    if settings.demo_router and demo_router is not None:
        if outcome.validator_result.accepted:
            demo_router.commit(proposal.proposal_id)
        else:
            demo_router.discard(proposal.proposal_id)

    # Persist updated session state and reject counter
    persistence_start = time.perf_counter()
    store.save_session(session, reject_counts=orchestrator._reject_counts)
    persistence_end = time.perf_counter()

    # Notify supervisors on automatic takeover
    if session.current_state == CallState.HUMAN_TAKEOVER:
        ws_manager = getattr(request.app.state, "supervisor_ws_manager", None)
        if ws_manager is not None:
            background_tasks.add_task(
                ws_manager.broadcast,
                {
                    "type": "takeover_alert",
                    "session_id": session_id,
                    "reason": "auto_escalation",
                    "timestamp": utc_now().isoformat(),
                },
            )

    voice_action = None
    rejection = None
    if action is not None:
        voice_action = VoiceActionSchema(
            action_type=action.action_type,
            template_id=action.template_id,
            allowed_variables=action.allowed_variables,
            interruptible=action.interruptible,
            timeout_ms=action.timeout_ms,
        )
    else:
        reason = (
            outcome.validator_result.reject_reason.value
            if outcome.validator_result.reject_reason
            else "rejected"
        )
        rejection = RejectionSchema(reason=reason)

    # Log latency breakdown for successful turn
    safety_ms = (safety_end - safety_start) * 1000 if (safety_start and safety_end) else 0
    rag_ms = (rag_end - rag_start) * 1000 if (rag_end and rag_start) else 0
    prompt_ms = (prompt_end - prompt_start) * 1000 if (prompt_end and prompt_start) else 0
    llm_ms = (llm_end - llm_start) * 1000 if (llm_end and llm_start) else 0
    validation_ms = (validation_end - validation_start) * 1000 if (validation_end and validation_start) else 0
    persistence_ms = (persistence_end - persistence_start) * 1000 if (persistence_end and persistence_start) else 0
    total_ms = (time.perf_counter() - turn_start) * 1000
    model_version = getattr(proposal, "model_version", "unknown") if proposal else "unknown"
    proposal_status = "accepted" if outcome.validator_result.accepted else "rejected"

    logger.info(
        f"turn_latency session_id={session_id} turn_id={turn_id} state={session.current_state.value} "
        f"safety_ms={safety_ms:.1f} rag_ms={rag_ms:.1f} prompt_build_ms={prompt_ms:.1f} "
        f"llm_ms={llm_ms:.1f} validation_ms={validation_ms:.1f} persistence_ms={persistence_ms:.1f} "
        f"total_turn_ms={total_ms:.1f} model_version={model_version} proposal_status={proposal_status} "
        f"rejection_reason={outcome.validator_result.reject_reason.value if outcome.validator_result.reject_reason else 'none'}"
    )

    return TurnResponse(
        session_id=session_id,
        turn_id=turn_id,
        current_state=session.current_state.value,
        voice_action=voice_action,
        rejection=rejection,
    )
