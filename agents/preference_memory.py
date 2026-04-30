"""User preference extraction for flight-based travel agent.

This module extracts preference hints from user queries related to flights
(seat preference, travel class) - hotel and transport not included in this release.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

SEAT_PREFERENCES = {
    "window": "window",
    "aisle": "aisle", 
    "middle": "middle",
}

TRAVEL_CLASSES = {
    "business": "business",
    "first": "first",
    "economy": "economy",
    "premium": "premium_economy",
    "premium economy": "premium_economy",
}


def extract_preferences_from_query(user_query: str) -> dict:
    """Extract preference hints from natural language query."""
    if not user_query:
        return {}
    
    prefs = {}
    q = user_query.lower()
    
    for keyword, seat in SEAT_PREFERENCES.items():
        if keyword in q:
            prefs["seat_preference"] = seat
            break
            
    for keyword, cls in TRAVEL_CLASSES.items():
        if keyword in q:
            prefs["travel_class"] = cls
            break
            
    return prefs


def parse_flight_preferences(flight_pref: Optional[str]) -> dict:
    """Parse flight_pref from intent into preference dict."""
    if not flight_pref:
        return {}
    return {"travel_class": flight_pref}


def merge_preferences(existing: dict, new: dict) -> dict:
    """Merge new preferences with existing, keeping established preferences."""
    merged = existing.copy()
    for key, value in new.items():
        if value and key not in merged:
            merged[key] = value
    return merged


class UserPreferenceStore:
    """Stores extracted preferences in session history meta field."""
    
    def __init__(self):
        self._preferences_cache = {}
    
    def extract_and_store(self, user_query: str, user_id: Optional[str], meta: dict) -> dict:
        """Extract preferences from query and add to session meta."""
        extracted = extract_preferences_from_query(user_query)
        if not extracted:
            return meta
            
        if user_id:
            cache_key = f"prefs_{user_id}"
            existing = self._preferences_cache.get(cache_key, {})
            merged = merge_preferences(existing, extracted)
            self._preferences_cache[cache_key] = merged
            meta = meta or {}
            meta["preferences"] = merged
            logger.debug(f"Stored preferences for {user_id}: {merged}")
        
        return meta
    
    def get_cached_preferences(self, user_id: str) -> Optional[dict]:
        """Get cached preferences for a user."""
        return self._preferences_cache.get(f"prefs_{user_id}")


preference_store = UserPreferenceStore()