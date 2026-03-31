# ADR 0002: Streaming Design and SSE Contract

## Status
Accepted

## Decision
The `/ask?stream=true` path uses Server-Sent Events (SSE) for incremental UI updates plus a final structured payload marker. The stream can emit typed SSE events (`reasoning_step`, `flights`, `weather`) during execution, followed by token chunks for narrative text, and must end with a `[DONE_JSON]{...}` payload that contains the final structured result or terminal error object. The frontend parser treats SSE frames as primary structured updates, accumulates text tokens separately, and transitions to non-stream fallback if stream initialization/activity fails within configured windows.

## Why This Was Chosen
We need a responsive UX that shows progress early while preserving a deterministic final state for rendering cards and actions. Pure token streaming alone is not enough for product UI because cards (best flight/weather/reasoning) need structured data before completion. The mixed contract (typed SSE + final JSON marker) gives:
- Early perceived responsiveness
- Stable final hydration contract
- Backward-compatible fallback to non-streaming `/ask`
- Clear testability around event ordering and completion behavior

The contract is intentionally simple and explicit so it is easy to debug in browser/network traces and straightforward to explain in interviews.

## Alternatives Considered
1. **WebSockets**  
More flexible bi-directional protocol, but unnecessary complexity for one-way server push in this use case.

2. **Token-only stream with no final structured marker**  
Simpler transport, but brittle UI state reconstruction and weaker reliability for post-stream rendering.

3. **Polling-only async jobs**  
Operationally simple, but worse UX latency and less real-time feel.

This decision optimizes for practical reliability, UX responsiveness, and low implementation overhead.
