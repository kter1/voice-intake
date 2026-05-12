# Voice Intake

[![CI](https://github.com/kter1/voice-intake/actions/workflows/ci.yml/badge.svg)](https://github.com/kter1/voice-intake/actions/workflows/ci.yml)

A supervised AI voice intake system for healthcare, demonstrating customer-pilot architecture patterns including protected HTTP and WebSocket surfaces, shared runtime clients, and full-payload audit-chain verification.

The system handles inbound calls end-to-end: STIR/SHAKEN attestation, split consent sequencing (recording + AI assistance), state-machine-driven field collection, a reject-only proposal validator that gates every LLM output, an append-only audit chain, real-time supervisor alerts, and a React supervisor dashboard.

> **Portfolio note:** Portfolio reference implementation demonstrating production-architecture
> and deployment-hardening patterns for a supervised AI voice system. Key areas: HMAC audit chain integrity, API/WebSocket
> auth, shared runtime lifecycle, state-machine-driven LLM gating, swappable LLM providers
> (remote and local AI providers), and a React supervisor dashboard. Backend and dashboard test suites pass on every CI run.

---

## Quickstart - AI Demo

### Primary: `make demo` - AI Demo (BYOK OpenAI-compatible remote LLM + seeded RAG)

This is the **primary AI demo path**. It demonstrates how a real AI intake system handles flexible caller input with LLM grounding and safety gates, all running at phone-call pace when configured with a low-latency provider.

Requires a user-supplied OpenAI-compatible provider key. Some providers may offer no-cost or free-tier options, but terms, billing requirements, model availability, and rate limits must be verified before use.

```bash
git clone https://github.com/kter1/voice-intake
cd voice-intake
cp .env.example .env
# Edit .env: set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
# Example: Groq llama-3.1-8b-instant is one OpenAI-compatible option
# Verify current terms, billing requirements, model availability, and rate limits before use
#
# Alternative: Google Gemini (verify no billing is enabled before use)
#   LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
#   LLM_CHAT_COMPLETIONS_PATH=/chat/completions
#   LLM_MODEL=gemini-2.5-flash-lite
make demo
```

Expected first response latency depends on provider, model, network, prompt size, and RAG retrieval. Use the backend `turn_latency` logs to verify actual performance. Provider availability and terms change; verify before use.

The demo starts the backend on `:8000` and the dashboard on `:5173`. Open **http://localhost:5173/demo**. The demo practice-policy corpus (`data/rag_seed/demo_practice/`) is seeded into a local ChromaDB instance on first start. Tap the microphone and speak, or type if your browser blocks microphone access.

> **What the demo does:** Run a real LLM with the demo prompt profile. Responses are generated through the LLM path and constrained by deterministic validation. The assistant is grounded in the seeded demo practice-policy corpus for questions about scheduling, insurance, and scope. ASR, eligibility, patient lookup, and Twilio validation remain mocked for local evaluation.

> **No scripted behavior:** `make demo` does not use `DemoRouter` or `MOCK_LLM`. If the LLM times out, the session escalates to human takeover through timeout handling.

> **Demo policy:** The demo forces `POLICY_PROFILE=demo`, which skips recording and AI-assistance consent gates because this portfolio demo is not actually recording calls. The default policy (used by telephony and any non-demo deployment) still requires both consents.

> **First start note:** The sentence-transformers embedding model (`all-MiniLM-L6-v2`) is downloaded on the first run if not already cached. Subsequent starts seed in under a second (idempotent - no duplicates).

> **Telephony note:** The Twilio telephony WebSocket path (`/ws/telephony/{session_id}`) remains on the default consent flow and is outside the public-demo scope.

---

## Manual QA Checklist

Start with `make demo` or `make demo-local` and open `http://localhost:5173/demo`. Expected responses:

- **"Who are you?"** -> AI explains it is an intake assistant for scheduling, not diagnosis. (Static meta-question fast path: deterministic response)
- **"Why do you need my name?"** -> AI explains; does not silently advance the flow.
- **"I do not feel comfortable giving personal information."** -> AI acknowledges and explains; does not bypass.
- **"Can you diagnose my knee pain?"** -> AI declines diagnosis; offers to schedule with a specialist.
- **"I am having chest pain."** -> Safety prerouter triggers `EMERGENCY_EXIT`; AI directs to emergency services.
- **"What is your cancellation policy?"** -> AI surfaces rescheduling guidance from the seeded corpus. (LLM + RAG path)
- **"Do you accept BlueCross?"** -> AI surfaces insurance reference from the seeded corpus. (LLM + RAG path)

Responses vary slightly on repeat for LLM-driven turns (scheduling, policy questions). Static opening meta-questions like "Who are you?" or "What can you do?" are answered via deterministic fast path and return the same response each time.

This demo is for portfolio review and local evaluation. It is not evidence of clinical deployment, payer connectivity, or regulatory approval.

---

## Internal Testing and Development

```bash
make dev        # deterministic scripted path (no LLM) - fast smoke/regression runs
make dev-ollama # real Ollama LLM, all other services mocked, no RAG seeding
make test       # full backend + dashboard test suite
```

`make dev` uses `MOCK_LLM=true` and `DEMO_ROUTER=true`. It exercises the scripted appointment-scheduling state machine for regression testing. It is **not** the AI demo experience - use `make demo` or `make demo-local` for that.

---

## What's implemented

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | Scripted call runner + fixtures | Complete |
| 2 | SQLAlchemy persistence (SQLite / PostgreSQL) | Complete |
| 3 | FastAPI HTTP API (sessions, turns, consent, audit, supervisor) | Complete |
| 4 | ChromaDB RAG + patient lookup + insurance eligibility | Complete |
| 5 | Remote and local LLM integration (tool-use) | Complete |
| 6 | Deepgram nova-3 streaming ASR via a pluggable adapter interface | Complete |
| 7 | React supervisor dashboard (Vite + TypeScript + Tailwind) | Complete |
| 8 | Twilio inbound webhook + Media Stream WebSocket | Complete |

**Verification:** `make test` runs the backend pytest suite plus dashboard Vitest regression tests. CI runs the full suite on every push to `main`.

---

## API overview

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/session/open` | Open a session, returns initial state |
| POST | `/session/{id}/turn` | Submit transcript or `audio_b64`, get `VoiceAction` |
| POST | `/session/{id}/consent` | Record recording / AI consent |
| GET  | `/session/{id}/audit` | Append-only audit trail |
| POST | `/session/{id}/supervisor/intervene` | Force takeover |
| GET  | `/sessions` | List all sessions |
| POST | `/telephony/twilio/voice` | Twilio inbound webhook, returns TwiML |
| POST | `/knowledge/upload` | Ingest PDF/DOCX/TXT into ChromaDB |
| WS   | `/ws/supervisors` | Real-time takeover alerts |
| WS   | `/ws/telephony/{id}` | Twilio Media Stream to Deepgram to turn pipeline |

Interactive docs are available at **http://localhost:8000/docs** only when `API_KEY` is blank.

---

## Configuration

All settings are read from environment variables or a `.env` file (see `.env.example`).

| Variable | Default | Notes |
|----------|---------|-------|
| `MOCK_LLM` | `true` in local workflows | Use scripted MockLLM instead of a real provider |
| `MOCK_ASR` | `true` in local workflows | Decode audio bytes as UTF-8 instead of calling Deepgram |
| `LLM_PROVIDER` | `remote` | `remote` or `ollama` |
| `LLM_BASE_URL` | - | Required when `LLM_PROVIDER=remote` |
| `LLM_API_KEY` | - | API key for the remote endpoint |
| `LLM_MODEL` | - | Model identifier for the remote endpoint |
| `LLM_CHAT_COMPLETIONS_PATH` | `/v1/chat/completions` | Chat-completions REST path; use `/chat/completions` for Gemini |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama endpoint |
| `OLLAMA_MODEL` | `llama3.1:8b` | Model name for Ollama |
| `DEEPGRAM_API_KEY` | - | Required when `MOCK_ASR=false` |
| `DATABASE_URL` | `sqlite:///./voice_intake.db` | PostgreSQL for production |
| `CHROMA_PATH` | `:memory:` | Persistent path for production RAG |
| `RAG_SEED_PATH` | - | Directory seeded into ChromaDB at startup when `MOCK_RAG=false` |
| `TWILIO_ACCOUNT_SID` | - | Required for inbound phone calls |
| `TWILIO_AUTH_TOKEN` | - | Required for inbound phone calls |
| `API_KEY` | - | Protects HTTP routes and disables docs when set |
| `STREAM_AUTH_SECRET` | - | Signs Twilio media-stream WebSocket tokens |
| `AUDIT_HMAC_SECRET` | - | Signs the per-session audit chain |
| `VITE_API_KEY` | - | Internal dashboard build-time API key forwarding |

### Internal dashboard trust model

`VITE_API_KEY` is compiled into the dashboard bundle. That is acceptable only for an internal operator dashboard on a trusted network or behind stronger perimeter controls. It is not a secure end-user authentication model for a public browser deployment.

---

## Architecture

```
 Inbound call (Twilio / browser demo)
        |
        v
 +--------------------------------------------------------------+
 |                    FastAPI HTTP / WS                         |
 |  POST /session/open  /turn  /consent  /supervisor            |
 |  WS /ws/telephony/{id}  /ws/supervisors                      |
 +-------------------------------+------------------------------+
                                 |
                                 v
 +--------------------------------------------------------------+
 |              VoiceIntakeOrchestrator                         |
 |                                                              |
 |  +------------------+   +--------------------+              |
 |  |  State Machine   |   |  Policy Engine     |              |
 |  |  (14 states,     |   |  consent seq.,     |              |
 |  |   deterministic) |   |  STIR/SHAKEN,      |              |
 |  +--------+---------+   |  fraud posture     |              |
 |           |             +--------+-----------+              |
 |           +------------------+--+                           |
 |                              v                              |
 |                 +------------------------+                  |
 |                 |   ProposalValidator    |                  |
 |                 |   (reject-only gate)   |                  |
 |                 +------------------------+                  |
 +--------------------------------------------------------------+
            |                         |
            v                         v
 +--------------------+    +-------------------------+
 |  SQLAuditStore     |    |  Supervisor Dashboard   |
 |  SQLite / PG       |    |  React + Vite + TS      |
 |  audit, sessions,  |    |  WS takeover alerts     |
 |  turns, consent    |    |  browser mic demo       |
 +--------------------+    +-------------------------+
            |
            v
 +--------------------------------------------------------------+
 |  Integrations (each toggleable with a MOCK_ flag)            |
 |                                                              |
 |  LLM:  Remote API or Ollama (tool-use, chat-completions)    |
 |  ASR:  Deepgram nova-3 streaming via pluggable adapter       |
 |  RAG:  ChromaDB + sentence-transformers (all-MiniLM-L6-v2)  |
 |  Tel:  Twilio Programmable Voice + Media Streams             |
 +--------------------------------------------------------------+
```

---

## Call flow

1. **Inbound call** arrives via Twilio (`POST /telephony/twilio/voice`) or the browser demo.
2. **Session opened** - STIR/SHAKEN attestation recorded, fraud risk calculated.
3. **Consent phase** - recording consent, then AI assistance consent (independently configurable).
4. **State-driven intake** - IDENTITY -> DEMOGRAPHICS -> INSURANCE -> REASON_FOR_VISIT -> READBACK -> DISPOSITION -> CLOSE.
5. **Every turn:** transcript -> RAG context lookup -> LLM `propose_action` tool call -> `ProposalValidator` gate -> state transition + `VoiceAction` emitted.
6. **Safety exits:** double-reject -> `HUMAN_TAKEOVER`; emergency keywords -> `EMERGENCY_EXIT`; supervisor override -> forced takeover.
7. **Real-time alerts:** WebSocket push to all connected supervisors on any takeover.

---

## Development

```bash
make install      # Python deps + npm install
make demo         # BYOK remote LLM demo + seeded RAG (requires .env)
make demo-local   # no-API-cost demo: local Ollama + seeded RAG (slower, no key needed)
make dev          # internal smoke: scripted/deterministic path (no LLM)
make dev-real     # real providers from .env (all services real)
make test         # backend + dashboard regression suites
make clean        # kill servers, remove __pycache__
```

Python 3.13+ and Node 18+ required.

---

## Design decisions

- **Model output is constrained:** LLM responses are treated as proposed actions and must pass deterministic validation before they affect session state.
- **State owns the workflow:** the intake flow is driven by explicit states, policy checks, and consent gates rather than free-form conversation alone.
- **Demo paths are separated:** `make demo` uses a configured remote provider and seeded RAG, while `make dev` remains deterministic for local regression testing.
- **External services are replaceable:** ASR, RAG, LLM, eligibility, patient lookup, and telephony integrations can run as real services or mocked local adapters.
- **Auditability is built in:** session events are persisted with an HMAC chain so audit integrity can be verified after the fact.
- **Human takeover is a first-class path:** supervisor intervention, emergency handling, and validation failures route to controlled takeover states instead of silent continuation.

---

## Deployment notes

1. Set `DATABASE_URL` to a PostgreSQL connection string.
2. Set `CHROMA_PATH` to a persistent directory.
3. Set `LLM_API_KEY`, `DEEPGRAM_API_KEY`, and `TWILIO_*` credentials.
4. Set `MOCK_LLM=false` and `MOCK_ASR=false`.
5. Set `API_KEY`, `STREAM_AUTH_SECRET`, and `AUDIT_HMAC_SECRET` for customer-pilot deployments.
6. Expose on HTTPS (required for Twilio webhook signature validation).
7. `uvicorn voice_intake.api.app:app --host 0.0.0.0 --port 8000 --workers 4`

For the dashboard: `npm run build` in `dashboard/`, then serve `dist/` from any static host.

## Customer-pilot architecture patterns

- shared retriever and LLM clients are created once in lifespan instead of per request
- HTTP routes require `X-API-Key` when `API_KEY` is set
- supervisor WebSocket requires `?api_key=...` for the internal dashboard
- telephony media-stream WebSocket uses short-lived signed tokens instead of an open URL
- audit events now carry a per-session HMAC chain with `/session/{id}/audit/verify`

## Out of scope

This project demonstrates customer-pilot architecture patterns, but it is not a complete clinical, HIPAA, or production deployment. A full HIPAA compliance program would additionally require controls such as IAM, encryption-at-rest guarantees, retention/deletion policy enforcement, incident runbooks, and third-party review.
