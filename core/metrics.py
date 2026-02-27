# core/metrics.py

import re
import logging
from prometheus_client import Counter, Histogram, Gauge
# --- dynamic increment helper (compatible with callers expecting `increment(name, tags={})`) ---
from typing import Dict, Tuple
from prometheus_client import Counter as PromCounter

logger = logging.getLogger(__name__)

# Airline API metrics
AIRLINE_RETRIES = Counter(
    "airline_retries_total",
    "Total airline API retries",
    ["reason"]  # timeout, http_429, http_5xx, network
)

AIRLINE_BREAKER_OPEN = Counter(
    "airline_breaker_open_total",
    "Total times airline circuit breaker opened"
)

# Number of attempts used per airline request (histogram)
AIRLINE_ATTEMPTS = Histogram(
    "airline_attempts_per_request",
    "Number of attempts used per airline API request"
)

# Weather API metrics
WEATHER_RETRIES = Counter(
    "weather_retries_total",
    "Total weather API retries",
    ["reason"]
)

WEATHER_BREAKER_OPEN = Counter(
    "weather_breaker_open_total",
    "Total times weather circuit breaker opened"
)

WEATHER_ATTEMPTS = Histogram(
    "weather_attempts_per_request",
    "Number of attempts used per weather API request"
)

# ----------------------------
# Request Counters
# ----------------------------

LLM_REQUESTS = Counter(
    "llm_requests_total",
    "Total LLM requests",
    ["provider", "status"]
)

TOOL_REQUESTS = Counter(
    "tool_requests_total",
    "Total tool API requests",
    ["tool", "status"]
)

# ----------------------------
# Latency Histograms
# ----------------------------

LLM_LATENCY = Histogram(
    "llm_request_latency_seconds",
    "LLM request latency",
    ["provider"]
)

TOOL_LATENCY = Histogram(
    "tool_request_latency_seconds",
    "Tool request latency",
    ["tool"]
)

# ----------------------------
# Stream Metrics
# ----------------------------

STREAM_REQUESTS = Counter(
    "stream_requests_total",
    "Total streaming requests",
    ["provider", "status"]  # status: started, success, error
)

STREAM_LATENCY = Histogram(
    "stream_latency_seconds",
    "End-to-end streaming latency (seconds)",
    ["provider"]
)

# ----------------------------
# Job Queue Metrics
# ----------------------------

JOB_QUEUE_SIZE = Gauge(
    "job_queue_size",
    "Number of jobs currently in the in-memory job queue"
)

# ----------------------------
# Circuit Breaker State
# ----------------------------

CIRCUIT_STATE = Gauge(
    "circuit_breaker_state",
    "Circuit breaker state (0=closed,1=open)",
    ["service"]
)

# ----------------------------
# LLM Metric Helper Functions
# ----------------------------

def increment_llm_success(provider: str) -> None:
    """
    Increment LLM success counter.
    """
    LLM_REQUESTS.labels(provider=provider, status="success").inc()


def increment_llm_failure(provider: str) -> None:
    """
    Increment LLM failure counter.
    """
    LLM_REQUESTS.labels(provider=provider, status="error").inc()


def increment_llm_cancelled(provider: str) -> None:
    """
    Increment LLM cancelled counter.
    Cancelled is tracked as a separate status.
    """
    LLM_REQUESTS.labels(provider=provider, status="cancelled").inc()


# ----------------------------
# Stream Metric Helper Functions
# ----------------------------

def record_stream_start(provider: str) -> None:
    """Record the start of a streaming request."""
    STREAM_REQUESTS.labels(provider=provider, status="started").inc()


def record_stream_success(provider: str, duration_sec: float) -> None:
    """Record a successful streaming completion with its duration."""
    STREAM_REQUESTS.labels(provider=provider, status="success").inc()
    STREAM_LATENCY.labels(provider=provider).observe(duration_sec)


def record_stream_failure(provider: str) -> None:
    """Record a failed streaming request."""
    STREAM_REQUESTS.labels(provider=provider, status="error").inc()


# --- sanitization helpers for dynamic metrics ---
def _sanitize_metric_name(name: str) -> str:
    """
    Convert any input into a valid Prometheus metric name:
    - allowed chars: [a-zA-Z0-9_:]
    - must not start with digit -> prefix with 'm_' if it does
    Replace invalid characters (dots, spaces, dashes, etc.) with '_'.
    """
    if not isinstance(name, str) or not name:
        return "metric_unknown"
    # replace any invalid char with underscore
    sanitized = re.sub(r'[^a-zA-Z0-9_:]', '_', name)
    # ensure it doesn't start with a digit
    if re.match(r'^[0-9]', sanitized):
        sanitized = 'm_' + sanitized
    return sanitized


def _sanitize_label_names(tags: Dict[str, str]) -> Dict[str, str]:
    """
    Prometheus label names must match [a-zA-Z_][a-zA-Z0-9_]*
    So convert invalid chars to '_' and ensure labels don't start with digit.
    """
    out = {}
    for k, v in (tags or {}).items():
        if not isinstance(k, str):
            continue
        nk = re.sub(r'[^a-zA-Z0-9_]', '_', k)
        if re.match(r'^[0-9]', nk):
            nk = 'l_' + nk
        # label values can be left as-is (they are strings)
        out[nk] = str(v)
    return out


# simple registry for dynamically-created counters keyed by (name, tuple(sorted_label_names))
_dynamic_counters: dict[Tuple[str, Tuple[str, ...]], PromCounter] = {}


def increment(name: str, tags: Dict[str, str] | None = None) -> None:
    """
    Dynamically create (if needed) and increment a Counter metric named `name`.
    This function sanitizes metric and label names and catches Prometheus errors so it
    never raises to the caller (logs warning instead).
    """
    try:
        tags = tags or {}
        # sanitize metric name & labels
        sanitized_name = _sanitize_metric_name(name)
        sanitized_tags = _sanitize_label_names(tags)

        label_names = tuple(sorted(sanitized_tags.keys()))
        key = (sanitized_name, label_names)

        # Create a Prometheus Counter for this metric name + label set if not present.
        if key not in _dynamic_counters:
            _dynamic_counters[key] = PromCounter(sanitized_name, f"Dynamic counter {sanitized_name}", list(label_names))

        counter = _dynamic_counters[key]
        if label_names:
            label_values = [sanitized_tags[k] for k in label_names]
            counter.labels(*label_values).inc()
        else:
            counter.inc()
    except Exception as e:
        # Do not let metric failures crash the app — just log.
        logger.warning(
            "metrics.increment failed (metric creation/labels). "
            "Original name=%s, sanitized=%s, tags=%s, error=%s",
            name, locals().get('sanitized_name', None), tags, str(e)
        )