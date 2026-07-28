import os
from contextlib import contextmanager
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/travel_planner")

@contextmanager
def get_checkpointer():
    """
    Context manager to yield a PostgresSaver checkpointer for the LangGraph agent.
    """
    with ConnectionPool(
        conninfo=DATABASE_URL,
        max_size=20,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
        },
    ) as pool:
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()
        yield checkpointer
