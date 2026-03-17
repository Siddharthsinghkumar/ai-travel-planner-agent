# core/iata_resolver.py
import re
import logging
from typing import Optional, List, Tuple
from airportsdata import load
from rapidfuzz import process, fuzz

AIRPORTS = load("IATA")

CITY_INDEX = {}
AIRPORT_INDEX = {}

# Build the indexes once at startup
for code, data in AIRPORTS.items():
    city = (data.get("city") or data.get("city_en") or data.get("location") or "").strip().lower()
    name = (data.get("name") or "").strip().lower()

    if city:
        CITY_INDEX.setdefault(city, []).append(code)
    if name:
        AIRPORT_INDEX[name] = code

logger = logging.getLogger(__name__)

def _top_two_matches(key_list: List[str], token: str) -> Tuple[Optional[Tuple[str, int]], Optional[Tuple[str, int]]]:
    """
    Return top two matches as tuples (key, score) or (None, None) if empty.
    """
    if not key_list:
        return None, None
    # get top 2
    results = process.extract(token, key_list, scorer=fuzz.WRatio, limit=2)
    if not results:
        return None, None
    first = (results[0][0], int(results[0][1]))
    second = (results[1][0], int(results[1][1])) if len(results) > 1 else (None, 0)
    return first, second

def _accept_match(best_score: int, second_score: int, min_score: int = 75, gap: int = 10) -> bool:
    """
    Decide whether to accept a match automatically.
    Accept if best_score >= min_score and (best_score - second_score >= gap or best_score >= 85).
    """
    if best_score < min_score:
        return False
    if best_score - second_score >= gap:
        return True
    # allow high absolute confidence even if gap small
    if best_score >= 85:
        return True
    return False

def _fuzzy_resolve_token(token: str) -> Optional[str]:
    """Try city keys then airport names with safe thresholds and tie-breaking."""
    # safety: skip tiny tokens (ambiguous)
    if not token or len(token) < 3:
        return None

    # 1) fuzzy against city names
    city_keys = list(CITY_INDEX.keys())
    best_city, second_city = _top_two_matches(city_keys, token)
    if best_city:
        best_key, best_score = best_city
        second_score = second_city[1] if second_city else 0
        if _accept_match(best_score, second_score, min_score=72, gap=10):
            logger.debug("Fuzzy city match: %s -> %s (score=%d, next=%d)", token, best_key, best_score, second_score)
            return CITY_INDEX[best_key][0]

    # 2) fuzzy against airport names
    name_keys = list(AIRPORT_INDEX.keys())
    best_name, second_name = _top_two_matches(name_keys, token)
    if best_name:
        best_key, best_score = best_name
        second_score = second_name[1] if second_name else 0
        if _accept_match(best_score, second_score, min_score=70, gap=10):
            logger.debug("Fuzzy airport-name match: %s -> %s (score=%d, next=%d)", token, best_key, best_score, second_score)
            return AIRPORT_INDEX[best_key]

    # 3) combined fallback (city + airport names). only if above failed:
    combined_keys = city_keys + name_keys
    best_comb, second_comb = _top_two_matches(combined_keys, token)
    if best_comb:
        best_key, best_score = best_comb
        second_score = second_comb[1] if second_comb else 0
        if _accept_match(best_score, second_score, min_score=70, gap=12):
            # prefer city mapping when possible
            if best_key in CITY_INDEX:
                return CITY_INDEX[best_key][0]
            if best_key in AIRPORT_INDEX:
                return AIRPORT_INDEX[best_key]
    return None

def normalize_text(text: str) -> str:
    """Strip punctuation and extra spaces."""
    return re.sub(r'[^a-z0-9\s]', '', text.lower()).strip()

def resolve_location(text: str) -> Optional[str]:
    if not text:
        return None

    token = normalize_text(text)
    if not token:
        return None

    # Known misspellings corrections (bypass fuzzy for common errors)
    _KNOWN_CORRECTIONS = {
        "dalhi": "DEL", "dilli": "DEL", "dehli": "DEL",
        "bombay": "BOM", "mumbay": "BOM",
        "banglore": "BLR", "bengaluru": "BLR",
    }
    for part in token.split():
        if part in _KNOWN_CORRECTIONS:
            return _KNOWN_CORRECTIONS[part]

    # direct IATA token quick path (single 3-letter token anywhere)
    for t in token.split():
        if len(t) == 3 and t.upper() in AIRPORTS:
            return t.upper()

    # Helper: test a candidate string against exact / substring / fuzzy
    def try_candidate(candidate: str) -> Optional[str]:
        cand = candidate.strip()
        if not cand:
            return None
        # exact city name
        if cand in CITY_INDEX:
            return CITY_INDEX[cand][0]
        # exact airport name
        if cand in AIRPORT_INDEX:
            return AIRPORT_INDEX[cand]
        # substring match (safe): e.g., 'new delhi' vs 'delhi'
        for city_key, codes in CITY_INDEX.items():
            if cand in city_key or city_key in cand:
                return codes[0]
        # fuzzy match against city keys (use WRatio)
        match = process.extractOne(cand, list(CITY_INDEX.keys()), scorer=fuzz.WRatio)
        if match and int(match[1]) >= 72:
            return CITY_INDEX[match[0]][0]
        # fuzzy match against airport names
        match2 = process.extractOne(cand, list(AIRPORT_INDEX.keys()), scorer=fuzz.WRatio)
        if match2 and int(match2[1]) >= 70:
            return AIRPORT_INDEX[match2[0]]
        return None

    parts = token.split()
    # 1) Try each single token (good for "dalhi" inside a sentence)
    for p in parts:
        if len(p) >= 3:
            r = try_candidate(p)
            if r:
                return r

    # 2) Try short n-grams (2 & 3 word combos) to capture multi-word city names
    n = len(parts)
    for size in (2, 3):
        if n < size:
            continue
        for i in range(0, n - size + 1):
            cand = " ".join(parts[i:i+size])
            r = try_candidate(cand)
            if r:
                return r

    # 3) As a last resort, try the whole phrase (existing behavior)
    whole = token.strip()
    r = try_candidate(whole)
    if r:
        return r

    return None

def is_iata_token(token: str) -> bool:
    if not token:
        return False
    t = token.strip()
    return len(t) == 3 and t.isalpha() and t.upper() in AIRPORTS