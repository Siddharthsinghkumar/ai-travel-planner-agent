# Monitoring Quickstart

This project exposes Prometheus metrics from the backend at `/metrics`.

## 1) Run backend and scrape metrics

1. Start the API normally (for example `uvicorn api.app:app --host 127.0.0.1 --port 8000`).
2. Verify scrape output:

```bash
curl -sS http://127.0.0.1:8000/metrics | head
```

## 2) Sample Prometheus config

Use [`monitoring/prometheus.yml`](./prometheus.yml) as a starter scrape config.
It also references [`monitoring/alerts.yml`](./alerts.yml) via `rule_files`.

To validate config locally:

```bash
promtool check config monitoring/prometheus.yml
promtool check rules monitoring/alerts.yml
```

## 3) Useful dashboard panels (Grafana)

- Request rate by route:
  - `sum by (route, method) (rate(http_requests_total[5m]))`
- API latency p50/p95:
  - `histogram_quantile(0.50, sum by (le, route, method) (rate(http_request_duration_seconds_bucket[5m])))`
  - `histogram_quantile(0.95, sum by (le, route, method) (rate(http_request_duration_seconds_bucket[5m])))`
- First-token latency p50/p95:
  - `histogram_quantile(0.50, sum by (le, provider) (rate(llm_first_token_latency_seconds_bucket[5m])))`
  - `histogram_quantile(0.95, sum by (le, provider) (rate(llm_first_token_latency_seconds_bucket[5m])))`
- Stream success/failure:
  - `sum by (provider, status) (rate(stream_requests_total[5m]))`
- Stream fallback rate:
  - `sum by (reason, provider) (rate(stream_fallback_total[5m]))`
- LLM route usage:
  - `sum by (mode, effective_mode, provider, stream) (rate(llm_route_usage_total[5m]))`

## 4) Notes

- Metrics use low-cardinality labels only (route/method/status class/provider/mode).
- `/metrics` requests are excluded from HTTP request metrics to avoid scrape skew.
- The streaming completion contract (`[DONE_JSON]`) remains unchanged.
- Grafana is not provisioned in this repo (no datasource/dashboard provisioning files). The queries above are ready to paste into a Grafana panel once Prometheus is configured.
