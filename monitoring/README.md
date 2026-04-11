# Monitoring Guide (Optional)

This repository includes an **optional local monitoring stack** for backend metrics:
- **Prometheus**: scrapes `/metrics` and lets you query time-series data.
- **Grafana**: visualizes those metrics with a pre-provisioned dashboard.

This is a local/operator aid, not a production observability platform rollout.

## What Exists In Repo

- Prometheus scrape config: `monitoring/prometheus.yml`
- Alert rules (optional): `monitoring/alerts.yml`
- Grafana datasource provisioning: `monitoring/grafana/provisioning/datasources/prometheus.yml`
- Grafana dashboard provisioning: `monitoring/grafana/provisioning/dashboards/dashboard.yml`
- Starter dashboard JSON: `monitoring/grafana/dashboards/llm-travel-hardening.json`
- Compose services (optional profile): `docker-compose.yml` (`prometheus`, `grafana`)

## 1) Start Monitoring

Run from repo root:

```bash
docker compose --profile monitoring up -d --build
```

This starts:
- API: `http://127.0.0.1:8000`
- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3000`

Default Grafana login:
- Username: `admin`
- Password: `admin`

## 2) Verify It Is Working

### A) Verify app metrics endpoint

```bash
curl -sS http://127.0.0.1:8000/metrics | head -n 40
```

You should see Prometheus text output with metric names like `http_requests_total`.

### B) Verify Prometheus scrape target

1. Open `http://127.0.0.1:9090/targets`
2. Confirm job `llm-travel-agent-api` is `UP`
3. If `DOWN`, open `http://127.0.0.1:9090/config` and confirm target is `api:8000`

### C) Verify Grafana datasource

1. Open `http://127.0.0.1:3000`
2. Log in (`admin` / `admin`)
3. Go to **Connections -> Data sources**
4. Confirm **Prometheus** datasource exists and is healthy

## 3) Use It (Operator Workflow)

### Prometheus checks

- Query UI: `http://127.0.0.1:9090/query`
- Useful quick checks:

```promql
up{job="llm-travel-agent-api"}

sum(rate(http_requests_total[5m]))

sum(rate(ask_admission_total{outcome="rejected_duplicate"}[5m]))
sum(rate(ask_admission_total{outcome="rejected_overload"}[5m]))

sum by (lookup_result, outcome) (increase(booking_handoff_consume_total[15m]))
sum by (component) (increase(retry_budget_exhausted_total[15m]))

sum by (service, event, reason_class) (increase(key_state_events_total[15m]))
sum by (provider, reason_class) (increase(provider_health_failures_total[15m]))
sum by (provider) (increase(provider_health_cooldown_skips_total[15m]))
sum by (transition) (increase(circuit_transitions_total[15m]))
```

### Grafana dashboard

1. Go to **Dashboards -> LLM Travel Agent -> LLM Travel Agent Hardening**
2. Review panels for hardening-focused signals:
   - Duplicate `/ask` rejections
   - Overload/backpressure rejections
   - Booking handoff consume outcomes
   - Retry budget exhaustion
   - Key state transitions
   - Provider health failures + cooldown skips
   - Circuit transitions

If dashboard is missing, check provisioning paths are mounted in Grafana container:
- `/etc/grafana/provisioning`
- `/var/lib/grafana/dashboards`

## 4) Alert Rules (Optional)

Alert rules in `monitoring/alerts.yml` are loaded by Prometheus and can be inspected at:
- `http://127.0.0.1:9090/rules`

These are local warning heuristics for validation/triage, not a complete on-call alerting setup.

## 5) Stop Monitoring

```bash
docker compose --profile monitoring down
```

To also delete Prometheus/Grafana persisted data volumes:

```bash
docker compose --profile monitoring down -v
```

## Known Limitations

- This setup is **single-node local** monitoring for this repo’s runtime.
- No distributed tracing/HA Prometheus/HA Grafana here.
- Alerting is basic and intentionally lightweight.
- If API traffic is low/idle, rate-based panels may look flat by design.
