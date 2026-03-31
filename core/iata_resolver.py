# core/iata_resolver.py
import os
import re
import logging
from typing import Optional, List, Tuple, Dict, Any
from airportsdata import load
from rapidfuzz import process, fuzz

AIRPORTS = load("IATA")

CITY_INDEX = {}
AIRPORT_INDEX = {}

# Build the indexes once at startup
for code, data in AIRPORTS.items():
    city_candidates = []
    for field in ("city", "city_en", "location"):
        value = str(data.get(field) or "").strip().lower()
        if value:
            city_candidates.append(value)
    name = (data.get("name") or "").strip().lower()

    for city in dict.fromkeys(city_candidates):
        CITY_INDEX.setdefault(city, []).append(code)
    if name:
        AIRPORT_INDEX[name] = code

logger = logging.getLogger(__name__)
_PREFERRED_COUNTRIES = tuple(
    c.strip().upper()
    for c in os.getenv("IATA_PREFERRED_COUNTRIES", "IN").split(",")
    if c.strip()
)

_LOCATION_ALIASES = {
    # Region-level and historical names that do not always appear as direct city keys.
    "goa": "GOI",
    "bombay": "BOM",
    "calcutta": "CCU",
}

_GENERIC_LOCATION_WORDS = {
    "city",
    "airport",
    "airports",
    "terminal",
    "international",
    "domestic",
}


def _strip_generic_location_words(text: str) -> str:
    """
    Remove generic location qualifiers from a normalized phrase so resolver
    matching is anchored on meaningful place tokens.
    """
    parts = [p for p in (text or "").split() if p not in _GENERIC_LOCATION_WORDS]
    return " ".join(parts).strip()


def _pick_city_code(codes: List[str]) -> Optional[str]:
    """Pick the most suitable code for a city with multiple airports."""
    if not codes:
        return None
    if not _PREFERRED_COUNTRIES:
        return codes[0]
    for country in _PREFERRED_COUNTRIES:
        for code in codes:
            info = AIRPORTS.get(code) or {}
            if str(info.get("country") or "").upper() == country:
                return code
    return codes[0]

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

def resolve_location_with_trace(text: str) -> Tuple[Optional[str], Dict[str, Any]]:
    trace: Dict[str, Any] = {
        "input": text,
        "normalized_input": None,
        "cleaned_input": None,
        "phrase_candidates": [],
        "selected_iata": None,
        "selected_city": None,
        "match_basis": "unresolved",
        "is_fuzzy": False,
        "confidence": None,
        "runner_up_confidence": None,
        "score_gap": None,
        "selected_candidate": None,
        "top_city_candidates": [],
        "top_airport_candidates": [],
    }
    if not text:
        trace["match_basis"] = "empty_input"
        return None, trace

    raw_text = text.strip()
    token = normalize_text(text)
    trace["normalized_input"] = token
    if not token:
        trace["match_basis"] = "empty_normalized_input"
        return None, trace

    cleaned_token = _strip_generic_location_words(token)
    trace["cleaned_input"] = cleaned_token
    phrase_candidates: List[str] = []
    for cand in (cleaned_token, token):
        if cand and cand not in phrase_candidates:
            phrase_candidates.append(cand)
    trace["phrase_candidates"] = list(phrase_candidates)

    def _finish(
        code: Optional[str],
        *,
        basis: str,
        selected_candidate: Optional[str] = None,
        is_fuzzy: bool = False,
        confidence: Optional[int] = None,
        runner_up_confidence: Optional[int] = None,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        trace["match_basis"] = basis
        trace["selected_candidate"] = selected_candidate
        trace["is_fuzzy"] = bool(is_fuzzy)
        trace["confidence"] = confidence
        trace["runner_up_confidence"] = runner_up_confidence
        if confidence is not None and runner_up_confidence is not None:
            trace["score_gap"] = confidence - runner_up_confidence
        if code and is_iata_token(code):
            selected = code.upper()
            trace["selected_iata"] = selected
            city = city_for_iata(selected)
            trace["selected_city"] = city
            return selected, trace
        return None, trace

    # Known misspellings corrections (bypass fuzzy for common errors)
    _KNOWN_CORRECTIONS = {
        "dalhi": "DEL", "dilli": "DEL", "dehli": "DEL",
        "bombay": "BOM", "mumbay": "BOM",
        "banglore": "BLR", "bengaluru": "BLR",
    }
    for phrase in phrase_candidates:
        alias_hit = _LOCATION_ALIASES.get(phrase)
        if alias_hit and is_iata_token(alias_hit):
            return _finish(alias_hit, basis="alias", selected_candidate=phrase)
        for part in phrase.split():
            correction = _KNOWN_CORRECTIONS.get(part)
            if correction and is_iata_token(correction):
                return _finish(correction, basis="known_correction", selected_candidate=part)

    # direct IATA token quick path, but only when user text explicitly used uppercase.
    for t in re.findall(r"\b([A-Z]{3})\b", raw_text):
        if t in AIRPORTS:
            return _finish(t, basis="explicit_iata", selected_candidate=t)

    def _is_preferred_code(code: Optional[str]) -> bool:
        if not code:
            return False
        info = AIRPORTS.get(code) or {}
        country = str(info.get("country") or "").upper()
        return bool(country and country in _PREFERRED_COUNTRIES)

    def _city_key_is_preferred(city_key: str) -> bool:
        selected = _pick_city_code(CITY_INDEX.get(city_key, []))
        return _is_preferred_code(selected)

    # Helper: test a candidate string against exact / substring / fuzzy
    def try_candidate(candidate: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        cand = candidate.strip()
        if not cand:
            return None, None
        # exact city name
        if cand in CITY_INDEX:
            selected = _pick_city_code(CITY_INDEX[cand])
            if selected:
                return selected, {
                    "basis": "exact_city",
                    "selected_candidate": cand,
                    "is_fuzzy": False,
                }
        # exact airport name
        if cand in AIRPORT_INDEX:
            return AIRPORT_INDEX[cand], {
                "basis": "exact_airport_name",
                "selected_candidate": cand,
                "is_fuzzy": False,
            }
        # Very short tokens are too ambiguous for substring/fuzzy matching.
        # Let callers handle explicit IATA-token intent in a dedicated layer.
        if len(cand) <= 3:
            return None, None
        # substring match (word-boundary-safe): e.g., 'new delhi' vs 'delhi'
        # Avoid arbitrary partial overlaps such as "kochi" matching "mangochi".
        for city_key, codes in CITY_INDEX.items():
            if re.search(rf"\b{re.escape(cand)}\b", city_key) or re.search(rf"\b{re.escape(city_key)}\b", cand):
                selected = _pick_city_code(codes)
                if selected:
                    return selected, {
                        "basis": "substring_city",
                        "selected_candidate": cand,
                        "is_fuzzy": False,
                    }
        # fuzzy match against city keys with tie-break telemetry.
        city_matches = process.extract(cand, list(CITY_INDEX.keys()), scorer=fuzz.WRatio, limit=8)
        if city_matches:
            trace["top_city_candidates"] = [
                {"candidate": m[0], "score": int(m[1]), "preferred_country": _city_key_is_preferred(m[0])}
                for m in city_matches[:3]
            ]
            best_key, best_score = city_matches[0][0], int(city_matches[0][1])
            best_pref = next((m for m in city_matches if _city_key_is_preferred(m[0])), None)
            use_preferred_tiebreak = False
            chosen_key, chosen_score = best_key, best_score

            # Generic safety rule: when a preferred-country city candidate is close in score,
            # prefer that city-level match over a non-preferred fuzzy winner.
            if best_pref and not _city_key_is_preferred(best_key):
                pref_key, pref_score = best_pref[0], int(best_pref[1])
                if pref_score >= 70 and (best_score - pref_score) <= 8:
                    chosen_key, chosen_score = pref_key, pref_score
                    use_preferred_tiebreak = True

            second_score = next((int(m[1]) for m in city_matches if m[0] != chosen_key), 0)
            accepted = (
                (chosen_score >= 70 and use_preferred_tiebreak)
                or _accept_match(chosen_score, second_score, min_score=72, gap=10)
            )
            if accepted:
                selected = _pick_city_code(CITY_INDEX.get(chosen_key, []))
                if selected:
                    return selected, {
                        "basis": "fuzzy_city_preferred_tiebreak" if use_preferred_tiebreak else "fuzzy_city",
                        "selected_candidate": chosen_key,
                        "is_fuzzy": True,
                        "confidence": int(chosen_score),
                        "runner_up_confidence": int(second_score),
                    }
        # fuzzy match against airport names with tie-break telemetry.
        best_airport, second_airport = _top_two_matches(list(AIRPORT_INDEX.keys()), cand)
        if best_airport:
            best_key, best_score = best_airport
            second_score = second_airport[1] if second_airport and second_airport[0] is not None else 0
            trace["top_airport_candidates"] = [
                {"candidate": best_key, "score": int(best_score)},
                {"candidate": second_airport[0], "score": int(second_score)} if second_airport and second_airport[0] is not None else None,
            ]
            trace["top_airport_candidates"] = [c for c in trace["top_airport_candidates"] if c]
            airport_min_score = 78 if len(cand.split()) == 1 else 70
            airport_gap = 12 if len(cand.split()) == 1 else 10
            if _accept_match(best_score, second_score, min_score=airport_min_score, gap=airport_gap):
                selected = AIRPORT_INDEX.get(best_key)
                if selected:
                    return selected, {
                        "basis": "fuzzy_airport_name",
                        "selected_candidate": best_key,
                        "is_fuzzy": True,
                        "confidence": int(best_score),
                        "runner_up_confidence": int(second_score),
                    }
        return None, None

    for phrase in phrase_candidates:
        # 1) Try the whole normalized phrase first.
        whole = phrase.strip()
        resolved, candidate_meta = try_candidate(whole)
        if resolved:
            return _finish(
                resolved,
                basis=candidate_meta.get("basis", "candidate_match"),
                selected_candidate=candidate_meta.get("selected_candidate", whole),
                is_fuzzy=bool(candidate_meta.get("is_fuzzy")),
                confidence=candidate_meta.get("confidence"),
                runner_up_confidence=candidate_meta.get("runner_up_confidence"),
            )

        parts = phrase.split()
        # 2) Try short n-grams (3 then 2 words) to capture multi-word city names.
        n = len(parts)
        for size in (3, 2):
            if n < size:
                continue
            for i in range(0, n - size + 1):
                cand = " ".join(parts[i:i + size])
                resolved, candidate_meta = try_candidate(cand)
                if resolved:
                    return _finish(
                        resolved,
                        basis=candidate_meta.get("basis", "candidate_ngram_match"),
                        selected_candidate=candidate_meta.get("selected_candidate", cand),
                        is_fuzzy=bool(candidate_meta.get("is_fuzzy")),
                        confidence=candidate_meta.get("confidence"),
                        runner_up_confidence=candidate_meta.get("runner_up_confidence"),
                    )

    # 3) As a last resort, try each token while skipping common travel noise words.
    noise = {
        "flight", "flights", "from", "to", "on", "at", "for", "via", "through",
        "stopover", "stop", "in", "return", "returning", "leaving", "departing",
        "coming", "back", "after", "before", "trip", "book", "booking", "ticket",
        "tickets", "cheapest", "cheap", "find", "and", "with", "under", "tomorrow",
        "today", "next", "this", "day", "days", "week", "weeks",
        *tuple(_GENERIC_LOCATION_WORDS),
    }
    for phrase in phrase_candidates:
        for p in phrase.split():
            if len(p) < 3 or p in noise or p.isdigit():
                continue
            resolved, candidate_meta = try_candidate(p)
            if resolved:
                return _finish(
                    resolved,
                    basis=candidate_meta.get("basis", "candidate_token_match"),
                    selected_candidate=candidate_meta.get("selected_candidate", p),
                    is_fuzzy=bool(candidate_meta.get("is_fuzzy")),
                    confidence=candidate_meta.get("confidence"),
                    runner_up_confidence=candidate_meta.get("runner_up_confidence"),
                )

    return _finish(None, basis="unresolved")


def resolve_location(text: str) -> Optional[str]:
    resolved, _trace = resolve_location_with_trace(text)
    return resolved

def is_iata_token(token: str) -> bool:
    if not token:
        return False
    t = token.strip()
    return len(t) == 3 and t.isalpha() and t.upper() in AIRPORTS


def city_for_iata(code: str) -> Optional[str]:
    """
    Resolve an IATA code to its city name when available.
    """
    if not code:
        return None
    normalized = code.strip().upper()
    if len(normalized) != 3:
        return None
    info = AIRPORTS.get(normalized)
    if not isinstance(info, dict):
        return None
    city = str(info.get("city") or info.get("city_en") or info.get("location") or "").strip()
    if not city:
        return None
    if city.islower():
        return city.title()
    return city


def label_for_iata(code: str) -> Optional[str]:
    """
    Resolve an IATA code to a display label in `City (IATA)` format.
    Falls back to the normalized IATA code when city metadata is unavailable.
    """
    if not code:
        return None
    normalized = code.strip().upper()
    if len(normalized) != 3 or normalized not in AIRPORTS:
        return None
    city = city_for_iata(normalized)
    if city:
        return f"{city} ({normalized})"
    return normalized
