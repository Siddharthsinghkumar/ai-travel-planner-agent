# core/exceptions.py
"""
Centralized exception definitions for the LLM Travel Agent system.

Why this exists:
- Avoid circular imports
- Provide a common base exception (LLMError)
- Allow structured catching at higher layers
- Make retry logic cleaner
- Improve testability
"""


# ============================================================
# Base Exceptions
# ============================================================

class TravelAgentError(Exception):
    """
    Root base exception for the entire application.
    All custom exceptions should inherit from this.
    """
    pass


class LLMError(TravelAgentError):
    """
    Base exception for all LLM-related failures
    (cloud or local).
    """
    pass


class ToolError(TravelAgentError):
    """
    Base exception for external tool/API failures
    (airline, weather, etc).
    """
    pass


# ============================================================
# Retry / Circuit Related
# ============================================================

class RetryableError(LLMError):
    """
    Raised when an operation is safe to retry.
    """
    pass


class CircuitBreakerError(TravelAgentError):
    """
    Raised when a circuit breaker blocks a request.
    """
    pass


# ============================================================
# Provider-Specific (Optional Expansion)
# ============================================================

class ProviderError(LLMError):
    """
    Raised when the underlying provider (OpenAI, Anthropic, Ollama)
    returns a malformed or invalid response.
    """
    pass


class StreamingError(LLMError):
    """
    Raised when a streaming response fails mid-stream.
    """
    pass
