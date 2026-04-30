import logging
import logging.config
import re
from typing import Any

from core.env_config import get_env_bool, get_env_str

_REDACTION_MASK = "***REDACTED***"
_SENSITIVE_KEY_TOKENS = (
    "token",
    "secret",
    "password",
    "authorization",
    "api_key",
    "apikey",
    "x-api-key",
    "x-admin-token",
    "admin_token",
    "cookie",
    "session",
    "passwd",
    "booking_token",
    "handoff_url",
    # Account/operational metadata that can leak quota/search state
    "account_data",
    "plan_name",
    "plan_searches_left",
    "searches_left",
    "total_searches_left",
    "details",
    "error_text",
    "key_value",
    "key_fp",
    "fingerprint",
    "raw_value",
    "response_text",
    "text_preview",
)
_SENSITIVE_ATTR_TOKENS = (
    "token",
    "secret",
    "password",
    "authorization",
    "api_key",
    "apikey",
    "cookie",
    "booking_token",
    "account_data",
    "plan_searches_left",
    "searches_left",
    "details",
    "error_text",
    "key_value",
)
_AUTH_BEARER_ASSIGNMENT_RE = re.compile(
    r"(?i)\bauthorization\b(\s*[:=]\s*)bearer\s+[^\s,;]+"
)
_KEY_VALUE_ASSIGNMENT_RE = re.compile(
    r'(?i)\b(api[_-]?key|token|secret|password|authorization|x-api-key|x-admin-token|admin_token|access_token|refresh_token|cookie|set-cookie)\b(\s*[:=]\s*)([^\s,;&]+)'
)
_JSON_SECRET_RE = re.compile(
    r'(?i)("?(?:api[_-]?key|token|secret|password|authorization|x-api-key|x-admin-token|admin_token|access_token|refresh_token|cookie|set-cookie)"?\s*:\s*)"([^"]*)"'
)
_QUERY_SECRET_RE = re.compile(
    r'(?i)([?&](?:api[_-]?key|token|secret|password|authorization|x-api-key|x-admin-token|access_token|refresh_token)=)([^&\s]+)'
)
_BEARER_TOKEN_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+\-/]+=*")
# Account/plan metadata patterns that can leak quota state in free-text log fields
_ACCOUNT_META_RE = re.compile(
    r'(?i)(plan_searches_left|searches_left|total_searches_left|plan_name|account_data|next_reset|reset_at)\s*[:=]\s*"?([^",}\s]+)'
)


def _parse_log_level() -> str:
    level = (get_env_str("LOG_LEVEL", "INFO") or "INFO").upper()
    return level if level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"} else "INFO"


def _parse_bool_env(name: str, default: bool = False) -> bool:
    return get_env_bool(name, default=default)


def _redact_text(text: str) -> str:
    value = str(text)
    value = _AUTH_BEARER_ASSIGNMENT_RE.sub(lambda m: f"Authorization{m.group(1)}{_REDACTION_MASK}", value)
    value = _KEY_VALUE_ASSIGNMENT_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}{_REDACTION_MASK}", value)
    value = _JSON_SECRET_RE.sub(lambda m: f'{m.group(1)}"{_REDACTION_MASK}"', value)
    value = _QUERY_SECRET_RE.sub(lambda m: f"{m.group(1)}{_REDACTION_MASK}", value)
    value = _BEARER_TOKEN_RE.sub(f"Bearer {_REDACTION_MASK}", value)
    value = _ACCOUNT_META_RE.sub(lambda m: f"{m.group(1)}={_REDACTION_MASK}", value)
    return value


def _key_looks_sensitive(key: Any) -> bool:
    key_l = str(key).lower()
    if any(token in key_l for token in _SENSITIVE_KEY_TOKENS):
        return True
    # Catch compound keys like "plan_searches_left", "account_data", etc.
    if any(token in key_l for token in ("plan_", "account_", "searches_", "reset_", "quota")):
        return True
    return False


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, dict):
        return {k: (_REDACTION_MASK if _key_looks_sensitive(k) else _redact_value(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item) for item in value)
    if isinstance(value, set):
        return {_redact_value(item) for item in value}
    return value


_STDLIB_LOG_ATTRS = frozenset({
    "name", "msg", "args", "created", "relativeCreated", "exc_info", "exc_text",
    "stack_info", "lineno", "funcName", "pathname", "filename", "module",
    "levelno", "levelname", "thread", "threadName", "process", "processName",
    "msecs", "message", "asctime", "taskName",
})


class StructuredExtraFormatter(logging.Formatter):
    """Appends structured extra fields as key=value pairs after the standard message."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in _STDLIB_LOG_ATTRS and not k.startswith("_")
        }
        if not extras:
            return base
        redacted_extras = _redact_value(extras)
        pairs = " ".join(f"{k}={v}" for k, v in sorted(redacted_extras.items()) if v is not None)
        return f"{base} | {pairs}" if pairs else base


class SensitiveDataRedactionFilter(logging.Filter):
    """Best-effort filter to prevent secret/token leakage in logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if record.args:
                try:
                    rendered = str(record.msg) % record.args
                except Exception:
                    record.msg = _redact_value(record.msg)
                    record.args = _redact_value(record.args)
                else:
                    record.msg = _redact_text(rendered)
                    record.args = ()
            else:
                record.msg = _redact_value(record.msg)
            # Redact exception info text
            if record.exc_info and record.exc_info[1] is not None:
                exc_text = str(record.exc_info[1])
                if exc_text:
                    record.exc_info = (record.exc_info[0], Exception(_redact_text(exc_text)), record.exc_info[2])
            if record.exc_text:
                record.exc_text = _redact_text(record.exc_text)
            for attr_name, attr_value in list(record.__dict__.items()):
                lower_attr = str(attr_name).lower()
                if any(token in lower_attr for token in _SENSITIVE_ATTR_TOKENS):
                    record.__dict__[attr_name] = _REDACTION_MASK
                elif isinstance(attr_value, str):
                    record.__dict__[attr_name] = _redact_text(attr_value)
                elif isinstance(attr_value, dict):
                    record.__dict__[attr_name] = _redact_value(attr_value)
        except Exception:
            return True
        return True
        return True


def setup_logging():
    log_level = _parse_log_level()
    enable_access_log = _parse_bool_env("ENABLE_UVICORN_ACCESS_LOG", default=False)

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "redact_sensitive": {
                "()": "core.logging_config.SensitiveDataRedactionFilter",
            }
        },
        "formatters": {
            "standard": {
                "()": "core.logging_config.StructuredExtraFormatter",
                "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "filters": ["redact_sensitive"],
                "stream": "ext://sys.stdout",
            }
        },
        "root": {
            "level": log_level,
            "handlers": ["console"],
        },
        "loggers": {
            "httpx": {"level": "WARNING"},
            "httpcore": {"level": "WARNING"},
            "urllib3": {"level": "WARNING"},
            "serpapi": {"level": "WARNING"},
            "uvicorn": {"level": log_level, "handlers": ["console"], "propagate": False},
            "uvicorn.error": {"level": log_level, "handlers": ["console"], "propagate": False},
            "uvicorn.access": {
                "level": "INFO" if enable_access_log else "WARNING",
                "handlers": ["console"],
                "propagate": False,
            },
        },
    })
