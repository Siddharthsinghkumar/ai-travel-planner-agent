# TECH-BRIEF-V2 — llm-travel-agent — Architecture & Capabilities

> Written 2026-07-28. Purpose: A dense technical summary of the updated system architecture, incorporating the LangGraph V2 migration, the React Vite SPA, and enhanced testing infrastructure.

## System in one paragraph

Python 3 / FastAPI monolith serving a multi-agent travel planner: NL query → regex+LLM intent parsing → LangGraph stateful orchestration (`/v2/ask`) orchestrating SerpAPI (Google Flights) and OpenWeather tools → multi-provider LLM router (Ollama local → Gemini → OpenAI; priority fallback, per-backend circuit breakers, atomic API-key rotation/cooldown) → explanation streamed over SSE (resilient `thread_id` chunking) to a React 19 + Vite (L'ÉVASION) frontend → booking handoff (SerpAPI token → provider checkout URL; NO payments). Sidecars: custom asyncio job queue w/ SQLAlchemy persistence, price tracker loop, HITL approval gate, local RAG, session memory w/ deterministic summarization. Persistence: Postgres Checkpointer (`langgraph-checkpoint-postgres`) and standard sync SQLAlchemy. Infra: Docker Compose. Tests: 228 fast (stub LLM, SQLite, auth off) + ungated slow suite + RAGAS evaluation harness (Merlin Opus / Local Stack Llama judges) + Playwright E2E. Logging: `structlog` enforcing JSON + Regex PII masking. 

## Architectural Shifts from V1

### 1. Agent Orchestration (LangGraph V2)
- **Previous:** Monolithic, hard-coded 6,907-line state machine with fragile string-sentinel stream protocols.
- **Current:** LangGraph `StateGraph` driven. State is defined in `TravelPlannerState` and checkpointed to Postgres. 
- **Resilience:** The frontend `useStreamingPlan.tsx` parses the initial SSE message for `thread_id`. If the network connection drops (e.g. during a server deploy), it seamlessly reconnects to `/v2/ask?thread_id=XYZ` resuming the identical graph execution state.

### 2. Frontend Modernization
- **Previous:** Multi-page HTML templates or early experimental React.
- **Current:** Single Omni-Box minimalist UI (L'ÉVASION) built on Vite, React 19, Tailwind, and TypeScript. Contains dedicated E2E testing flows via Playwright.

### 3. Observability & Gating
- **Logging:** Structlog intercepts all logging events. A strict `pii_processor` scrubs emails, phone numbers, and passport IDs dynamically.
- **Metrics:** Prometheus exporter now binds to port `8765`, recording precise LLM token usage.
- **CI/CD:** Github Actions pipeline now enforces Ruff, Fast Suite, Slow Suite (PG), RAGAS evaluations, and Playwright before auto-merging `dev` into `main`.
