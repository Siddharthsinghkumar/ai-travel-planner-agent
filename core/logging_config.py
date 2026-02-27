# core/logging_config.py

import logging
import json
from datetime import datetime, UTC
from core.request_context import get_request_id


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": get_request_id(),
        }

        # If extra fields were passed
        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                if key not in log_record and key not in (
                    "args", "msg", "levelname", "levelno",
                    "pathname", "filename", "module",
                    "exc_info", "exc_text", "stack_info",
                    "lineno", "funcName", "created",
                    "msecs", "relativeCreated", "thread",
                    "threadName", "processName", "process"
                ):
                    log_record[key] = value
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


def setup_logging():
    # Prevent sensitive data from being logged by HTTP libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)