from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from voice_intake.api.middleware import APIKeyMiddleware
from voice_intake.api.routers import (
    audit,
    capabilities,
    consent,
    knowledge,
    sessions,
    supervisor,
    telephony,
    turns,
)
from voice_intake.api.security import (
    resolve_runtime_secrets,
    verify_api_key,
    verify_stream_token,
)
from voice_intake.api.ws import SupervisorWSManager
from voice_intake.config import Settings, get_settings
from voice_intake.db import SQLAuditStore, create_tables, get_engine, make_session_factory
from voice_intake.knowledge.retrieval import KnowledgeRetriever
from voice_intake.llm.client import LLMClient
from voice_intake.llm.dispatch import propose_next_action
from voice_intake.llm.ollama_client import OllamaClient

_log = logging.getLogger(__name__)


def _open_chroma(settings: Settings):  # type: ignore[no-untyped-def]
    import chromadb  # type: ignore[import]

    if settings.chroma_path == ":memory:":
        return chromadb.EphemeralClient()
    return chromadb.PersistentClient(path=settings.chroma_path)


def _build_llm_client(settings: Settings) -> LLMClient | OllamaClient | None:
    if settings.mock_llm:
        return None
    if settings.llm_provider == "ollama":
        return OllamaClient(
            settings.ollama_base_url,
            settings.ollama_model,
            settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
    # remote - only build if base_url and model are configured
    if not settings.llm_base_url or not settings.llm_model:
        return None
    return LLMClient(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout_seconds,
        chat_completions_path=settings.llm_chat_completions_path,
        max_retries=settings.llm_max_retries,
    )


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings: Settings = app.state.settings
    engine = get_engine(settings.database_url)
    create_tables(engine)
    app.state.engine = engine
    app.state.session_factory = make_session_factory(engine)
    app.state.chroma_client = _open_chroma(settings)
    if settings.mock_rag:
        # Skip retriever construction entirely so SentenceTransformer never loads.
        # `get_knowledge_retriever` also short-circuits to None when mock_rag is set,
        # so the dependency won't re-instantiate from chroma_client.
        app.state.retriever = None
    else:
        app.state.retriever = KnowledgeRetriever(
            app.state.chroma_client,
            embed_fn=getattr(app.state, "embed_fn", None),
        )
        if settings.rag_seed_path:
            seed_path = Path(settings.rag_seed_path)
            if seed_path.exists():
                from voice_intake.knowledge.seed import seed_directory

                try:
                    embed_fn = (
                        app.state.retriever._embed
                        if app.state.retriever is not None
                        else None
                    )
                    results = seed_directory(
                        seed_root=seed_path,
                        practice_id="demo",
                        category="seed",
                        chroma_client=app.state.chroma_client,
                        embed_fn=embed_fn,
                    )
                    seeded = sum(1 for r in results if not r.skipped)
                    skipped = sum(1 for r in results if r.skipped)
                    chunks = sum(r.chunk_count for r in results)
                    _log.info(
                        "RAG seed complete from %s: seeded=%d skipped=%d chunks=%d",
                        seed_path,
                        seeded,
                        skipped,
                        chunks,
                    )
                except Exception:
                    _log.error("RAG seed failed from %s", seed_path, exc_info=True)
            else:
                _log.error("RAG seed path does not exist: %s", seed_path)
    from voice_intake.safety_prerouter import SafetyPreRouter

    app.state.safety_pre_router = SafetyPreRouter()
    app.state.llm_client = _build_llm_client(settings)
    if settings.demo_router:
        from voice_intake.demo_router import DemoRouter

        app.state.demo_router = DemoRouter(
            session_factory=app.state.session_factory,
            audit_secret=app.state.audit_secret,
        )
    else:
        app.state.demo_router = None
    try:
        yield
    finally:
        llm_client = getattr(app.state, "llm_client", None)
        if llm_client is not None:
            await llm_client.aclose()
        engine.dispose()


def create_app(settings: Settings | None = None) -> FastAPI:
    runtime_settings = settings or get_settings()
    audit_secret, stream_secret = resolve_runtime_secrets(runtime_settings)
    app = FastAPI(
        title="Voice Intake API",
        description="Supervised AI voice intake system for healthcare.",
        version="0.4.0",
        docs_url=None if runtime_settings.api_key else "/docs",
        redoc_url=None if runtime_settings.api_key else "/redoc",
        openapi_url=None if runtime_settings.api_key else "/openapi.json",
        lifespan=_lifespan,
    )
    app.state.settings = runtime_settings
    app.state.audit_secret = audit_secret
    app.state.stream_secret = stream_secret
    app.state.supervisor_ws_manager = SupervisorWSManager()

    app.add_middleware(APIKeyMiddleware, api_key=runtime_settings.api_key)
    cors_origins = ["http://localhost:5173", "http://localhost:4173"] + list(
        getattr(runtime_settings, "cors_origins", [])
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    app.include_router(sessions.router)
    app.include_router(turns.router)
    app.include_router(consent.router)
    app.include_router(audit.router)
    app.include_router(supervisor.router)
    app.include_router(knowledge.router)
    app.include_router(telephony.router)
    app.include_router(capabilities.router)

    @app.websocket("/ws/supervisors")
    async def supervisors_ws(websocket: WebSocket) -> None:
        manager: SupervisorWSManager = websocket.app.state.supervisor_ws_manager
        settings: Settings = websocket.app.state.settings
        provided = websocket.query_params.get("api_key")
        if not verify_api_key(provided, settings.api_key):
            await websocket.close(code=4401)
            return
        await manager.connect(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            manager.disconnect(websocket)

    @app.websocket("/ws/telephony/{session_id}")
    async def telephony_media_stream(websocket: WebSocket, session_id: str) -> None:
        token = websocket.query_params.get("token", "")
        if not verify_stream_token(token, session_id, websocket.app.state.stream_secret):
            await websocket.close(code=4401)
            return
        await _handle_telephony_ws(websocket, session_id)

    return app


async def _handle_telephony_ws(websocket: WebSocket, session_id: str) -> None:
    from voice_intake.models import CallState, Speaker, VoiceTurn, utc_now
    from voice_intake.normalization import extract_field_candidates, normalize_turn
    from voice_intake.orchestrator import VoiceIntakeOrchestrator
    from voice_intake.policy import PolicyEngine, PolicyProfile
    from voice_intake.prompting import build_model_prompt
    from voice_intake.telephony.media_stream import MediaStreamHandler
    from voice_intake.templates import DEFAULT_TEMPLATE_BUNDLE
    from voice_intake.validator import ProposalValidator

    settings: Settings = websocket.app.state.settings
    store = SQLAuditStore(
        websocket.app.state.session_factory,
        hmac_secret=websocket.app.state.audit_secret,
    )
    session = store.load_session(session_id)
    if session is None:
        await websocket.close(code=1008)
        return

    await websocket.accept()

    ws_manager: SupervisorWSManager = websocket.app.state.supervisor_ws_manager
    twilio_client = _make_twilio_client(settings)
    session_ref = [session]

    async def on_transcript(transcript: str, is_final: bool) -> None:
        if not is_final or not transcript.strip():
            return

        reloaded = store.load_session(session_id)
        if reloaded is None:
            return
        session_ref[0] = reloaded
        current_session = session_ref[0]

        profile = PolicyProfile()
        policy_engine = PolicyEngine(profile)
        validator = ProposalValidator(DEFAULT_TEMPLATE_BUNDLE, policy_engine)
        orch = VoiceIntakeOrchestrator(
            policy_profile=profile,
            validator=validator,
            audit_store=store,
            policy_engine=policy_engine,
        )
        orch._reject_counts = store.load_reject_counts(session_id)

        turn_id = str(uuid4())
        ts = utc_now()
        turn = VoiceTurn(
            turn_id=turn_id,
            speaker=Speaker.CALLER,
            audio_ref=None,
            retention_policy_id="standard",
            transcript=transcript,
            partial_or_final="final",
            asr_confidence=0.9,
            barge_in=False,
            start_ts=ts,
            end_ts=ts,
        )
        store.record_turn(session_id, turn)

        knowledge_chunks = None
        retriever = getattr(websocket.app.state, "retriever", None)
        if retriever is not None:
            chunks = retriever.retrieve(
                f"{current_session.current_state.value} {transcript[:200]}", top_k=3
            )
            if chunks:
                knowledge_chunks = [
                    {
                        "text": c.text,
                        "source": c.source,
                        "category": c.category,
                        "score": c.score,
                    }
                    for c in chunks
                ]

        prompt = build_model_prompt(
            current_session,
            [turn],
            extract_field_candidates(normalize_turn(turn)),
            list(validator.template_bundle.templates.values()),
            knowledge_chunks=knowledge_chunks,
        )

        proposal = await propose_next_action(
            prompt,
            source_turn_id=turn_id,
            settings=settings,
            llm_client=getattr(websocket.app.state, "llm_client", None),
            safety_pre_router=getattr(websocket.app.state, "safety_pre_router", None),
        )
        if proposal is None:
            # LLM timeout escalates to HUMAN_TAKEOVER. handle_model_timeout sets
            # current_state=HUMAN_TAKEOVER, mode=HUMAN_TAKEOVER, and records a chained
            # MODEL_TIMEOUT audit event.
            orch.handle_model_timeout(current_session, buffered_audio_ref=None)
            store.save_session(current_session, reject_counts=orch._reject_counts)
            session_ref[0] = current_session
            await ws_manager.broadcast(
                {
                    "type": "takeover_alert",
                    "session_id": session_id,
                    "reason": "llm_timeout",
                    "timestamp": utc_now().isoformat(),
                }
            )
            return

        _outcome, action = orch.handle_model_proposal(current_session, proposal)
        store.save_session(current_session, reject_counts=orch._reject_counts)
        session_ref[0] = current_session

        if current_session.current_state == CallState.HUMAN_TAKEOVER:
            await ws_manager.broadcast(
                {
                    "type": "takeover_alert",
                    "session_id": session_id,
                    "reason": "auto_escalation",
                    "timestamp": utc_now().isoformat(),
                }
            )

        if twilio_client and handler.call_sid:
            _inject_twiml(
                twilio_client,
                handler.call_sid,
                action,
                current_session.current_state,
                settings.human_agent_queue,
            )

    handler = MediaStreamHandler(session_id=session_id, on_transcript=on_transcript)
    dg_key = settings.deepgram_api_key if not settings.mock_asr else None
    await handler.handle(websocket, api_key=dg_key, mock_asr=settings.mock_asr)


def _make_twilio_client(settings: object) -> object | None:
    account_sid = getattr(settings, "twilio_account_sid", "")
    auth_token = getattr(settings, "twilio_auth_token", "")
    if not account_sid or not auth_token:
        return None
    try:
        from twilio.rest import Client  # type: ignore[import]

        return Client(account_sid, auth_token)
    except Exception:
        _log.warning("Could not create Twilio REST client - missing SDK?")
        return None


def _inject_twiml(
    twilio_client: object,
    call_sid: str,
    action: object | None,
    current_state: object,
    queue_name: str,
) -> None:
    from voice_intake.models import CallState
    from voice_intake.telephony.twilio_adapter import build_dial_twiml, build_say_twiml
    from voice_intake.templates import DEFAULT_TEMPLATE_BUNDLE

    try:
        if current_state == CallState.HUMAN_TAKEOVER:
            twiml = build_dial_twiml(queue_name=queue_name)
        elif action is not None:
            template_id = getattr(action, "template_id", "")
            template = DEFAULT_TEMPLATE_BUNDLE.template_for(template_id)
            if template is None:
                return
            text = template.content
            variables: dict[str, str] = getattr(action, "allowed_variables", {}) or {}
            for key, val in variables.items():
                text = text.replace(f"{{{{{key}}}}}", str(val))
            text = text.strip()
            if not text:
                return
            twiml = build_say_twiml(text)
        else:
            return
        twilio_client.calls(call_sid).update(twiml=twiml)  # type: ignore[union-attr]
    except Exception:
        _log.exception("Failed to inject TwiML via REST call_sid=%s", call_sid)


app = create_app()
