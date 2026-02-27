# gemini_multikey_9_3_helper_script.py
# Simple test stub for Gemini-like client. NOT a production client.
import itertools

class GeminiClient:
    def __init__(self, keys=None, per_key_limit=5):
        # example keys list, replace with real keys or configure externally
        self.keys = keys or ["fake-key-1", "fake-key-2"]
        self.per_key_limit = per_key_limit
        self._counters = {k: 0 for k in self.keys}
        self._iter = itertools.cycle(self.keys)

    def _rotate_key(self):
        # naive rotation: pick next key
        return next(self._iter)

    def generate(self, prompt, model=None, max_output_tokens=128, temperature=0.0):
        # Simulate rate limit for the currently selected key sometimes
        key = self._rotate_key()
        self._counters[key] += 1
        # Simulate rate-limit by raising occasionally if counter exceeds per_key_limit
        if self._counters[key] > self.per_key_limit:
            raise RuntimeError("GeminiKeyRateLimit: key exhausted for today")
        # Make a deterministic fake response
        return f"[gemini:{key}] response for: {prompt[:200]}"

    def generate_single(self, prompt, max_output_tokens=128, temperature=0.0):
        return self.generate(prompt, max_output_tokens=max_output_tokens, temperature=temperature)

    def close(self):
        pass

# module-level convenience API (used by cloud_llm adapter)
def generate(prompt, max_output_tokens=128, temperature=0.0, model=None):
    # instantiate a simple singleton for module calls
    global _inst
    try:
        _inst
    except NameError:
        _inst = GeminiClient()
    return _inst.generate(prompt, model=model, max_output_tokens=max_output_tokens, temperature=temperature)