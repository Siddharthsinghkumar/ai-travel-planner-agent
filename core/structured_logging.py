import structlog
import logging
import re
from typing import Any
from core.logging_config import _redact_value

_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
_PASSPORT_RE = re.compile(r"\b[A-Z]{1,2}[0-9]{7,8}\b")
_PII_REDACTION_MASK = "***PII_REDACTED***"

def _redact_pii_text(text: str) -> str:
    value = str(text)
    value = _EMAIL_RE.sub(_PII_REDACTION_MASK, value)
    value = _PHONE_RE.sub(_PII_REDACTION_MASK, value)
    value = _PASSPORT_RE.sub(_PII_REDACTION_MASK, value)
    return value

def _redact_pii(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_pii_text(value)
    if isinstance(value, dict):
        return {k: _redact_pii(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_pii(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_pii(item) for item in value)
    if isinstance(value, set):
        return {_redact_pii(item) for item in value}
    return value

def redact_processor(logger, log_method, event_dict):
    """Structlog processor to apply both secret and PII redaction."""
    # Redact the event message
    if "event" in event_dict:
        event_dict["event"] = _redact_pii(_redact_value(event_dict["event"]))
    
    # Redact all kwargs
    for k, v in list(event_dict.items()):
        if k != "event":
            event_dict[k] = _redact_pii(_redact_value(v))
            
    return event_dict

def setup_structlog():
    # 1. Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.contextvars.merge_contextvars, # To support thread_id binding
            redact_processor,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # 2. Configure standard library logging to use structlog
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    
    # We respect the LOG_LEVEL env from before
    from core.env_config import get_env_str
    level_str = (get_env_str("LOG_LEVEL", "INFO") or "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    root_logger.setLevel(level)
