"""Shared error classification utilities.

Provides unified error/failure classification across services.
Each caller can use these directly or subclass for service-specific behavior.
"""

import asyncio
import httpx
from typing import Optional, Union


def classify_error_type(
    error: Union[Exception, int, str],
    categories: Optional[dict] = None,
) -> str:
    """
    Generic error classifier that dispatches based on error type.
    
    Args:
        error: Exception, HTTP status code (int), or error text (str)
        categories: Optional custom category mapping
        
    Returns:
        Category string like "timeout", "network", "auth_failure", etc.
    """
    if isinstance(error, int):
        return classify_http_status(error)
    
    if isinstance(error, Exception):
        # Check common exception types first
        if isinstance(error, (httpx.TimeoutException, asyncio.TimeoutError)):
            return "timeout"
        if isinstance(error, (httpx.ConnectError, httpx.NetworkError)):
            return "network"
        if isinstance(error, ValueError):
            return "response_parse"
            
        # Fall back to text classification
        return classify_text(str(error), categories)
    
    # If it's a string, treat as error text
    if isinstance(error, str):
        return classify_text(error, categories)
    
    return "unexpected"


def classify_http_status(status_code: Optional[int]) -> str:
    """Classify HTTP status code into category."""
    if status_code is None:
        return "unknown_failure"
    status_code = int(status_code)
    
    if status_code == 401 or status_code == 403:
        return "auth_failure"
    if status_code == 429:
        return "rate_limit"
    if 400 <= status_code < 500:
        return "request_failure"
    if status_code >= 500:
        return "upstream_failure"
    if status_code < 400:
        return "success"
    return "unknown_failure"


def classify_text(
    error_text: str,
    categories: Optional[dict] = None,
) -> str:
    """
    Classify error text into category using common patterns.
    
    Args:
        error_text: Error message text
        categories: Optional custom {pattern: category} mapping
        
    Returns:
        Category string
    """
    text = (error_text or "").lower()
    
    # Custom categories override defaults
    if categories:
        for pattern, category in categories.items():
            if pattern.lower() in text:
                return category
    
    # Common patterns
    if "timeout" in text:
        return "timeout"
    if "no available keys" in text or "no usable keys" in text:
        return "no_active_key"
    if "429" in text or "rate limit" in text or "too many requests" in text:
        return "rate_limit"
    if "403" in text or "401" in text or "unauthorized" in text:
        return "auth_failure"
    if "500" in text or "502" in text or "503" in text or "upstream" in text:
        return "upstream_failure"
    if "network" in text or "connection" in text:
        return "network"
    
    return "unexpected"


# Common category patterns for different services
SERPAPI_CATEGORIES = {
    "no available keys for service": "no_active_key",
    "rate limit": "rate_limit",
    "unauthorized": "auth_failure",
    "quota": "quota_exhausted",
}

PROVIDER_CATEGORIES = {
    "no available keys for service": "no_active_key",
    "rate limit": "rate_limit",
    "unauthorized": "auth_failure",
    "circuit open": "circuit_open",
    "timeout": "timeout",
}

BOOKING_OPTIONS_CATEGORIES = {
    "no available keys for service": "no_active_key",
    "rate limit": "provider_rate_limited",
    "unauthorized": "provider_auth",
}