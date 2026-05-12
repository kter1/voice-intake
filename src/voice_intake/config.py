from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    database_url: str = "sqlite:///./voice_intake.db"
    llm_provider: Literal["remote", "ollama"] = "remote"
    llm_prompt_profile: Literal["default", "demo", "recruiter_demo"] = "default"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    deepgram_api_key: str = ""
    mock_llm: bool = False
    mock_asr: bool = False
    # When true, skip RAG retriever construction entirely so the embedding model
    # never loads. Used by `make dev` and `make test` for the zero-network promise.
    mock_rag: bool = False
    # When true, routes browser /turn calls through DemoRouter instead of MockLLM
    # or a remote/Ollama LLM.  Set DEMO_ROUTER=true to enable the scheduling demo
    # flow.  Has no effect on the telephony WebSocket path.
    demo_router: bool = False
    # Selects the call-flow policy profile.  "demo" skips recording / AI-assistance
    # consent gates and enters at OPENING (used by the public scheduling demo,
    # which is not actually recording calls).  Independent of demo_router.
    policy_profile: Literal["default", "demo"] = "default"
    llm_chat_completions_path: str = "/v1/chat/completions"
    llm_timeout_seconds: float = 8.0
    llm_max_retries: int = 3
    api_key: str = ""
    stream_auth_secret: str = ""
    audit_hmac_secret: str = ""
    debug: bool = False
    log_level: str = "INFO"
    # Knowledge & RAG (Phase 4)
    chroma_path: str = ":memory:"
    mock_eligibility: bool = False
    mock_patient_lookup: bool = False
    patient_lookup_csv: str = ""
    # CORS (comma-separated origins added to the defaults localhost:5173 + 4173)
    cors_origins: list[str] = []
    # Telephony - Twilio (Phase 8)
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""
    human_agent_queue: str = "human-agent-queue"
    # Set to True to skip Twilio signature validation (tests / local dev)
    mock_twilio_validate: bool = False
    # When non-empty and mock_rag=false, seed this directory into ChromaDB at startup.
    rag_seed_path: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
