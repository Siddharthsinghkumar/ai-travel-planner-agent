# agents/database.py  — REPLACE top part up to Base declaration

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Text,
    TIMESTAMP,
    JSON,
    String,
    DateTime,
    Boolean,
    UniqueConstraint,
    inspect,
    text,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from dotenv import load_dotenv
from pathlib import Path
import logging
import os
from datetime import datetime
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)

# Do NOT rely on import-time environment being fully configured.
# We'll attempt to load a .env file if present, but prefer real environment variables.
def _ensure_env_loaded():
    # load project .env if it exists, but do not override existing env vars
    load_dotenv(ROOT_DIR / ".env", override=False)

def _build_engine(database_url: Optional[str] = None):
    if database_url is None:
        _ensure_env_loaded()
        database_url = os.getenv("DATABASE_URL")

    is_testing = os.getenv("TESTING", "false").lower() in ("1", "true", "yes", "on")
    testing_use_persistent_db = os.getenv("TESTING_USE_PERSISTENT_DB", "false").lower() in ("1", "true", "yes", "on")
    if is_testing and not testing_use_persistent_db:
        # Use in-memory SQLite for pytest/import safety.
        # StaticPool ensures all sessions share one in-memory DB.
        return create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )

    if not database_url:
        database_url = "sqlite:///./local.db"
    return create_engine(database_url, pool_pre_ping=True)

# Lazy module-level engine/session storage
_engine = None
_SessionLocal = None

def init_engine_and_session(database_url: Optional[str] = None):
    global _engine, _SessionLocal
    if _engine is None:
        _engine = _build_engine(database_url)
        _SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)
    return _engine, _SessionLocal

def get_engine():
    engine, _ = init_engine_and_session()
    return engine

def get_session():
    _, SessionLocal = init_engine_and_session()
    return SessionLocal()

Base = declarative_base()

# Keep your ORM model here unchanged...
class SessionHistory(Base):
    __tablename__ = "session_history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=True)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    user_query = Column(Text, nullable=False)
    agent_reasoning = Column(JSON, nullable=True)
    tool_output = Column(JSON, nullable=True)
    final_response = Column(Text, nullable=True)
    meta = Column(JSON, nullable=True)


class ProviderKeyState(Base):
    """
    Durable provider key-state records for reconciliation-aware key rotation/exhaustion.
    Source of truth for SerpAPI provider state across process restarts.
    """
    __tablename__ = "provider_key_states"
    __table_args__ = (
        UniqueConstraint("provider", "key_name_fingerprint", name="uq_provider_key_name_fp"),
    )

    id = Column(Integer, primary_key=True, index=True)
    provider = Column(String(50), nullable=False, index=True)
    key_name_fingerprint = Column(String(64), nullable=False, index=True)
    key_value_fingerprint = Column(String(64), nullable=False, index=True)
    is_exhausted = Column(Boolean, nullable=False, default=False)
    exhausted_until = Column(DateTime, nullable=True)
    retry_after = Column(DateTime, nullable=True)
    searches_left = Column(Integer, nullable=True)
    last_checked_at = Column(DateTime, nullable=True)
    last_used_at = Column(DateTime, nullable=True)
    expected_reset_basis = Column(String(64), nullable=True)
    expected_reset_at = Column(DateTime, nullable=True)
    last_error = Column(Text, nullable=True)
    last_reason = Column(String(128), nullable=True)
    failure_classification = Column(String(64), nullable=True)
    state_meta = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class ProviderStateOverride(Base):
    """
    Durable manual operator overrides for provider/key/account/project state.
    Supports provider-aware semantics (key scope for SerpAPI, provider/project scopes for Gemini, etc.).
    """
    __tablename__ = "provider_state_overrides"

    id = Column(Integer, primary_key=True, index=True)
    provider = Column(String(50), nullable=False, index=True)
    scope_type = Column(String(32), nullable=False, index=True)  # key | provider_account | project
    scope_identifier = Column(String(128), nullable=True, index=True)
    # Explicit key-scope bindings (fingerprints only; never raw secrets).
    key_name_fingerprint = Column(String(64), nullable=True, index=True)
    key_value_fingerprint = Column(String(64), nullable=True, index=True)
    override_type = Column(String(64), nullable=False, index=True)
    active_until = Column(DateTime, nullable=True)
    note = Column(Text, nullable=True)
    is_enabled = Column(Boolean, nullable=False, default=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

def _ensure_provider_key_state_columns(engine) -> None:
    try:
        inspector = inspect(engine)
        if not inspector.has_table("provider_key_states"):
            return
        existing = {str(col.get("name") or "") for col in inspector.get_columns("provider_key_states")}
        dialect_name = str(getattr(getattr(engine, "dialect", None), "name", "") or "").strip().lower()
        datetime_sql = "TIMESTAMP"
        json_sql = "JSON"
        if dialect_name in {"postgres", "postgresql"}:
            datetime_sql = "TIMESTAMP WITHOUT TIME ZONE"
            json_sql = "JSONB"
        alters = []
        if "exhausted_until" not in existing:
            alters.append(f"ALTER TABLE provider_key_states ADD COLUMN exhausted_until {datetime_sql}")
        if "retry_after" not in existing:
            alters.append(f"ALTER TABLE provider_key_states ADD COLUMN retry_after {datetime_sql}")
        if "last_used_at" not in existing:
            alters.append(f"ALTER TABLE provider_key_states ADD COLUMN last_used_at {datetime_sql}")
        if "state_meta" not in existing:
            alters.append(f"ALTER TABLE provider_key_states ADD COLUMN state_meta {json_sql}")
        if not alters:
            return
        with engine.begin() as conn:
            for stmt in alters:
                conn.execute(text(stmt))
    except Exception:
        logger.exception("provider_key_state_schema_ensure_failed")
        raise


def _ensure_provider_override_columns(engine) -> None:
    try:
        inspector = inspect(engine)
        if not inspector.has_table("provider_state_overrides"):
            return
        existing = {str(col.get("name") or "") for col in inspector.get_columns("provider_state_overrides")}
        alters = []
        if "key_name_fingerprint" not in existing:
            alters.append("ALTER TABLE provider_state_overrides ADD COLUMN key_name_fingerprint VARCHAR(64)")
        if "key_value_fingerprint" not in existing:
            alters.append("ALTER TABLE provider_state_overrides ADD COLUMN key_value_fingerprint VARCHAR(64)")
        if not alters:
            return
        with engine.begin() as conn:
            for stmt in alters:
                conn.execute(text(stmt))
    except Exception:
        logger.exception("provider_override_schema_ensure_failed")
        raise

def init_db():
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    _ensure_provider_key_state_columns(engine)
    _ensure_provider_override_columns(engine)


# Backwards compatibility for modules that import SessionLocal
# Many callers do: from agents.database import SessionLocal; s = SessionLocal()
# Provide a callable that behaves the same (returns a session instance).
def SessionLocal():
    """
    Backwards-compatible callable. Calling SessionLocal() -> returns a DB Session.
    """
    return get_session()
