import logging
from agents.ollama_llm import generate as ollama_generate, OllamaError

USE_CLOUD_FALLBACK = True

try:
    from agents.cloud_llm import generate as cloud_generate
except ImportError:
    cloud_generate = None

def generate(prompt: str, system: str = "") -> str:
    """
    Local-first LLM routing.
    1. Try Ollama
    2. Fallback to cloud if enabled
    """
    try:
        return ollama_generate(prompt, system)
    except OllamaError as e:
        logging.warning(f"Ollama failed: {e}")

        if USE_CLOUD_FALLBACK and cloud_generate:
            logging.warning("Falling back to cloud LLM")
            return cloud_generate(prompt, system)

        raise RuntimeError("All LLM backends failed")
