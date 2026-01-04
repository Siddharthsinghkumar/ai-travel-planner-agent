import requests
import os

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11450pi/generate")
OLLAMA_MODEL = os.getenv("openhermes")

class OllamaError(Exception):
    pass

def generate(prompt: str, system: str = "") -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"{system}\n\n{prompt}".strip(),
        "stream": False
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["response"]
    except Exception as e:
        raise OllamaError(str(e))
