# ADR 0001: LLM Routing and Fallback Strategy

## Status
Accepted

## Decision
The backend uses a router-based LLM strategy with explicit runtime modes (`ollama_only`, `cloud_only`, `ollama_first`, `cloud_first`), short availability probes, backend ordering, and bounded fallback behavior. For each request, the router picks an effective mode based on requested mode plus current backend availability. It applies per-backend timeouts and records route/failure metrics. If the first backend fails before response/first token, the router can fall back to the next backend. For cloud routing, provider fallback is controlled by mode-aware policy flags.

## Why This Was Chosen
This project needs practical reliability without hiding failures. A mode-based router gives deterministic behavior during demos, local development, and degraded cloud conditions. It supports clear tradeoffs:
- Cost/control by preferring local models when appropriate
- Availability by failing over to cloud when local is down (or vice versa)
- Observability by emitting route/fallback/failure metrics and structured reasons

This approach keeps behavior explicit for interviews and debugging: we can explain exactly why a backend was selected and why fallback happened.

## Alternatives Considered
1. **Single-provider only**  
Simpler implementation, but weak resilience and poor story when that provider is unavailable or rate-limited.

2. **Always race providers in parallel**  
Potentially lower latency in some cases, but higher cost, harder cancellation semantics, and noisier operational behavior for this project scale.

3. **Implicit “auto” routing only**  
Less configuration burden, but weak control for testing and harder to reason about in failures.

The selected design balances reliability, controllability, and implementation complexity for a production-style portfolio project.
