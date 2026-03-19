import logging
import os
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

TRUTHY_VALUES = {"1", "true", "yes", "on"}
FALSY_VALUES = {"0", "false", "no", "off", ""}


def get_env_str(name: str, default: Optional[str] = None, strip: bool = True) -> Optional[str]:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip() if strip else raw


def is_env_set(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and value.strip() != ""


def get_env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    normalized = raw.strip().lower()
    if normalized in TRUTHY_VALUES:
        return True
    if normalized in FALSY_VALUES:
        return False

    logger.warning("Invalid boolean env %s=%r; using default=%s", name, raw, default)
    return default


def get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        logger.warning("Invalid integer env %s=%r; using default=%s", name, raw, default)
        return default


def get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except (TypeError, ValueError):
        logger.warning("Invalid float env %s=%r; using default=%s", name, raw, default)
        return default


def parse_csv_env(name: str, default: Optional[Iterable[str]] = None) -> list[str]:
    raw = get_env_str(name, default=None)
    if raw is None:
        return [item.strip() for item in (default or []) if str(item).strip()]
    return [part.strip() for part in raw.split(",") if part.strip()]
