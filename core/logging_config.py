# core/logging_config.py

import logging
import logging.config
from core.env_config import get_env_bool, get_env_str


def _parse_log_level() -> str:
    level = (get_env_str("LOG_LEVEL", "INFO") or "INFO").upper()
    return level if level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"} else "INFO"


def _parse_bool_env(name: str, default: bool = False) -> bool:
    return get_env_bool(name, default=default)


def setup_logging():
    log_level = _parse_log_level()
    enable_access_log = _parse_bool_env("ENABLE_UVICORN_ACCESS_LOG", default=False)

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
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
            "uvicorn": {"level": log_level, "handlers": ["console"], "propagate": False},
            "uvicorn.error": {"level": log_level, "handlers": ["console"], "propagate": False},
            "uvicorn.access": {
                "level": "INFO" if enable_access_log else "WARNING",
                "handlers": ["console"],
                "propagate": False,
            },
            "gunicorn": {"level": log_level, "handlers": ["console"], "propagate": False},
            "gunicorn.error": {"level": log_level, "handlers": ["console"], "propagate": False},
            "gunicorn.access": {
                "level": "INFO" if enable_access_log else "WARNING",
                "handlers": ["console"],
                "propagate": False,
            },
        },
    })
