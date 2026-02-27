# core/llm_mode.py
import os

async def get_llm_mode_and_priority():
    """
    Returns:
        mode: "local" | "cloud" | "hybrid"
        priority: "local-first" | "cloud-first"
    """
    mode = os.getenv("LLM_MODE", "hybrid")
    priority = os.getenv("LLM_PRIORITY", "local-first")
    return mode, priority