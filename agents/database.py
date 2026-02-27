# database.py
from sqlalchemy import create_engine, Column, Integer, Text, TIMESTAMP, JSON, String
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from pathlib import Path
import os
from datetime import datetime

#  Force-load .env from project root
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./local.db"


engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

class SessionHistory(Base):
    __tablename__ = "session_history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=True)            # optional
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    user_query = Column(Text, nullable=False)
    agent_reasoning = Column(JSON, nullable=True)      # store LLM prompt & chain of thought
    tool_output = Column(JSON, nullable=True)          # raw tool responses (flight lists, weather)
    final_response = Column(Text, nullable=True)       # final LLM text returned to user
    meta = Column(JSON, nullable=True)                 # any other metadata

def init_db():
    Base.metadata.create_all(bind=engine)