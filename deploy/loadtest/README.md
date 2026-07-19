# k6 Load Test — llm-travel-agent

## What this tests

Ramping load: 0→10 VUs→0 over 2 minutes, weighted ~90% `GET /health` / ~10% `POST /ask`.
Uses the EXACT `AskRequest` fields (origin, destination, date, user_query, trip_type) — no
invented fields, per `extra="forbid"`.

## Live command (§6 M2-T11)

```bash
k6 run -e BASE_URL=https://<domain> deploy/loadtest/travel-agent.js
```

If prod requires auth on /ask and AUTH_DISABLE is unset, add the token:

```bash
k6 run -e BASE_URL=https://<domain> -e AUTH_TOKEN=<token> deploy/loadtest/travel-agent.js
```

**The citable capacity number is produced in §6 M2-T11 against the live box; any local run is
a BASELINE only.**

## Thresholds

| Threshold | Value |
|---|---|
| `http_req_failed` | < 1% |
| `http_req_duration` p95 | < 3000ms |
| `checks` | > 99% |

## Inspection

```bash
k6 inspect deploy/loadtest/travel-agent.js
```
