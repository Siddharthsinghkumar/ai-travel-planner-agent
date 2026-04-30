"""Service-layer exception exports.

Provides a stable interface between API layer and tool implementations.
Tool exceptions should not be imported directly from tools at the API layer.
"""

from tools.airline_api import AirlineAPIError
from tools.weather_api import WeatherAPIError

__all__ = [
    "AirlineAPIError",
    "WeatherAPIError",
]


def get_consume_post_handoff_artifact():
    """Lazy import for consume_post_handoff_artifact_with_diagnostics."""
    from tools.booking_handoff import consume_post_handoff_artifact_with_diagnostics

    return consume_post_handoff_artifact_with_diagnostics