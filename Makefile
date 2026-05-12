# voice-intake - developer workflow
# Usage:
#   make demo          BYOK OpenAI-compatible remote LLM demo + seeded RAG (requires .env)
#   make dev           internal smoke mode (scripted/deterministic) - regression testing only
#   make dev-real      real APIs from .env  (set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL)
#   make dev-ollama    local Ollama LLM + mock services
#   make test          backend + dashboard tests
#   make install       install Python + Node dependencies only
#   make clean         kill background servers, remove caches

.PHONY: install demo demo-local dev dev-real dev-ollama test clean

ifdef API_KEY
  BACKEND_AUTH_ENV := API_KEY=$(API_KEY)
  DASHBOARD_AUTH_ENV := VITE_API_KEY=$(API_KEY)
else
  BACKEND_AUTH_ENV :=
  DASHBOARD_AUTH_ENV :=
endif

BACKEND_DEV_BASE := MOCK_ASR=true MOCK_RAG=true MOCK_ELIGIBILITY=true MOCK_PATIENT_LOOKUP=true \
	MOCK_TWILIO_VALIDATE=true DATABASE_URL=sqlite:///./voice_intake.db \
	CHROMA_PATH=':memory:'

# Demo base: real RAG (no MOCK_RAG, no CHROMA_PATH override - demo sets those directly).
BACKEND_DEMO_BASE := MOCK_ASR=true MOCK_ELIGIBILITY=true MOCK_PATIENT_LOOKUP=true \
	MOCK_TWILIO_VALIDATE=true DATABASE_URL=sqlite:///./voice_intake.db

# demo (BYOK remote LLM + seeded RAG; requires .env)
# Bring your own OpenAI-compatible provider key. Some providers may offer
# no-cost or free-tier options; verify current terms, billing requirements,
# model availability, and rate limits before use.
demo:
	@test -f .env || \
		(echo ""; \
		 echo "  ERROR: .env not found."; \
		 echo "  Run:  cp .env.example .env"; \
		 echo "  Then set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL in .env."; \
		 echo "  Use a provider/key you have intentionally configured."; \
		 echo "  Alternatively, run: make demo-local  (no API key needed)"; \
		 echo ""; \
		 exit 1)
	@. ./.env && test -n "$$LLM_BASE_URL" || \
		(echo "ERROR: LLM_BASE_URL not set in .env - set your OpenAI-compatible provider endpoint." && exit 1)
	@. ./.env && test -n "$$LLM_API_KEY" || \
		(echo "ERROR: LLM_API_KEY not set in .env - set your provider API key." && exit 1)
	@. ./.env && test -n "$$LLM_MODEL" || \
		(echo "ERROR: LLM_MODEL not set in .env - e.g. llama-3.1-8b-instant for Groq." && exit 1)
	$(MAKE) install
	@echo ""
	@echo "  Starting Voice Intake AI DEMO (remote low-latency provider)"
	@echo "  LLM: configured provider (LLM_BASE_URL / LLM_MODEL from .env)"
	@echo "  RAG: demo practice-policy corpus seeded into .chroma-demo/"
	@echo "  Backend  ->  http://localhost:8000"
	@echo "  Dashboard -> http://localhost:5173/demo"
	@echo ""
	@echo "  Note: Verify provider terms before use. Check turn_latency logs for actual latency."
	@echo "  Open http://localhost:5173/demo in Chrome."
	@echo ""
	env LLM_PROVIDER=remote MOCK_LLM=false DEMO_ROUTER=false \
		MOCK_RAG=false CHROMA_PATH='.chroma-demo/' RAG_SEED_PATH='data/rag_seed/demo_practice' \
		LLM_PROMPT_PROFILE=demo POLICY_PROFILE=demo \
		$(BACKEND_DEMO_BASE) $(BACKEND_AUTH_ENV) \
	uvicorn voice_intake.api.app:app \
		--host 0.0.0.0 --port 8000 --reload \
		--log-level info &
	env $(DASHBOARD_AUTH_ENV) npm --prefix dashboard run dev

# demo local (no API cost fallback local Ollama + seeded RAG)
# Guaranteed no API cost. Slower: first response typically 5-10 s (model warm-up).
demo-local: install
	@command -v ollama >/dev/null 2>&1 || \
		(echo "ERROR: install Ollama first, then run: ollama pull llama3.1:8b"; exit 1)
	@echo ""
	@echo "  Starting Voice Intake DEMO-LOCAL (local Ollama)"
	@echo "  WARNING:  This is an offline engineering fallback, not the public demo path."
	@echo "  LLM: local Ollama (llama3.1:8b) - no API cost; responses 5-10s+ (not phone-call latency)"
	@echo "  RAG: demo practice-policy corpus seeded into .chroma-demo/"
	@echo "  Backend  ->  http://localhost:8000"
	@echo "  Dashboard -> http://localhost:5173/demo"
	@echo ""
	@echo "  For the public demo, use:  make demo  (requires provider config in .env)"
	@echo "  Open http://localhost:5173/demo in Chrome."
	@echo ""
	env LLM_PROVIDER=ollama LLM_PROMPT_PROFILE=demo MOCK_LLM=false DEMO_ROUTER=false \
		POLICY_PROFILE=demo \
		MOCK_RAG=false CHROMA_PATH='.chroma-demo/' RAG_SEED_PATH='data/rag_seed/demo_practice' \
		$(BACKEND_DEMO_BASE) $(BACKEND_AUTH_ENV) \
	uvicorn voice_intake.api.app:app \
		--host 0.0.0.0 --port 8000 --reload \
		--log-level info &
	env $(DASHBOARD_AUTH_ENV) npm --prefix dashboard run dev

# install
install:
	@echo "Installing Python dependencies..."
	pip3 install -e ".[dev]" --break-system-packages -q
	@echo "Installing dashboard dependencies..."
	cd dashboard && npm install --silent

# dev (internal smoke scripted/deterministic, no real LLM)
# NOTE: For portfolio demo use `make demo`. This target is for regression testing only.
dev: install
	@echo ""
	@echo "  Starting Voice Intake in MOCK mode"
	@echo "  Backend  ->  http://localhost:8000"
	@echo "  Dashboard -> http://localhost:5173/demo"
	@echo ""
	@echo "  Open http://localhost:5173/demo in Chrome. The demo starts automatically - speak or type."
	@echo ""
	env MOCK_LLM=true DEMO_ROUTER=true $(BACKEND_DEV_BASE) $(BACKEND_AUTH_ENV) \
	uvicorn voice_intake.api.app:app \
		--host 0.0.0.0 --port 8000 --reload \
		--log-level warning &
	env $(DASHBOARD_AUTH_ENV) npm --prefix dashboard run dev

# dev real (reads .env real remote AI endpoint + Deepgram)
dev-real: install
	@test -f .env || { echo "ERROR: .env not found. Run: cp .env.example .env"; exit 1; }
	@echo ""
	@echo "  Starting Voice Intake with REAL APIs (reading .env)"
	@echo "  Backend  ->  http://localhost:8000"
	@echo "  Dashboard -> http://localhost:5173/demo"
	@echo ""
	env $(BACKEND_AUTH_ENV) uvicorn voice_intake.api.app:app \
		--host 0.0.0.0 --port 8000 --reload \
		--log-level info &
	env $(DASHBOARD_AUTH_ENV) npm --prefix dashboard run dev

# dev ollama (local Ollama model, no remote API key needed)
dev-ollama: install
	@command -v ollama >/dev/null 2>&1 || \
		(echo "ERROR: install Ollama first, then run: ollama pull llama3.1:8b"; exit 1)
	@echo ""
	@echo "  Starting Voice Intake with local Ollama"
	@echo "  Backend  ->  http://localhost:8000"
	@echo "  Dashboard -> http://localhost:5173/demo"
	@echo ""
	env LLM_PROVIDER=ollama MOCK_LLM=false $(BACKEND_DEV_BASE) $(BACKEND_AUTH_ENV) \
	uvicorn voice_intake.api.app:app \
		--host 0.0.0.0 --port 8000 --reload \
		--log-level warning &
	env $(DASHBOARD_AUTH_ENV) npm --prefix dashboard run dev

# test
test:
	PYTHONPATH=src \
	MOCK_LLM=true \
	MOCK_ASR=true \
	MOCK_RAG=true \
	MOCK_ELIGIBILITY=true \
	MOCK_PATIENT_LOOKUP=true \
	MOCK_TWILIO_VALIDATE=true \
	DATABASE_URL='sqlite:///:memory:' \
	CHROMA_PATH=':memory:' \
	python3 -m pytest tests/ -v
	npm --prefix dashboard test -- --run

# clean
clean:
	-pkill -f "uvicorn voice_intake" 2>/dev/null
	-pkill -f "vite" 2>/dev/null
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	@echo "Done."
