# agents/database.py  — REPLACE top part up to Base declaration

from sqlalchemy import create_engine, Column, Integer, Text, TIMESTAMP, JSON, String
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from pathlib import Path
import os
from datetime import datetime
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent.parent

# Do NOT rely on import-time environment being fully configured.
# We'll attempt to load a .env file if present, but prefer real environment variables.
def _ensure_env_loaded():
    # load project .env if it exists, but do not override existing env vars
    load_dotenv(ROOT_DIR / ".env", override=False)

def _build_engine(database_url: Optional[str] = None):
    if database_url is None:
        _ensure_env_loaded()
        database_url = os.getenv("DATABASE_URL")
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

def init_db():
    engine = get_engine()
    Base.metadata.create_all(bind=engine)


# Backwards compatibility for modules that import SessionLocal
# Many callers do: from agents.database import SessionLocal; s = SessionLocal()
# Provide a callable that behaves the same (returns a session instance).
def SessionLocal():
    """
    Backwards-compatible callable. Calling SessionLocal() -> returns a DB Session.
    """
    return get_session()