"""Intent parser - extracts structured data from natural language queries."""

import re
from typing import Any, Dict, Optional, Tuple

from core.iata_resolver import resolve_location
from core.iata_resolver import is_iata_token


WORD_TO_NUM = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
    'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11,
    'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
    'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
    'twenty': 20, 'thirty': 30, 'fortnight': 14,
}


def _sanitize_iata_code(value: Optional[str]) -> Optional[str]:
    """Ensure value is a valid 3-letter IATA code (or None)."""
    if not value:
        return None
    code = str(value).strip().upper()
    if len(code) == 3 and code.isalpha() and is_iata_token(code):
        return code
    return None


def _clean_fragment(text: Optional[str]) -> Optional[str]:
    """Clean a route fragment."""
    if text is None:
        return None
    cleaned = re.sub(r"\s+", " ", text).strip(" ,.-")
    for _ in range(3):
        next_cleaned = cleaned
        next_cleaned = re.sub(
            r"^(?:please\s+|kindly\s+)?(?:find|search|show|get|book)(?:\s+me)?\s+",
            "",
            next_cleaned,
            flags=re.IGNORECASE,
        )
        next_cleaned = re.sub(
            r"^(?:a|an|the|flight|flights|trip|from|to|round[\s-]*trip|one[\s-]*way|return(?:\s+flight)?|multi[\s-]*city)\s+",
            "",
            next_cleaned,
            flags=re.IGNORECASE,
        )
        if next_cleaned == cleaned:
            break
        cleaned = next_cleaned
    cleaned = re.sub(r"\s+(?:on|for|at)\s+\d{4}-\d{2}-\d{2}\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\s+(?:leave|leav(?:e|ing)|depart(?:ing)?|return(?:ing)?)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned or None


def extract_stopover(query_text: str) -> Dict[str, Optional[str]]:
    """
    Robust extraction for 'via' stopover phrases.
    Returns dict with keys: origin_text, destination_text, via_text
    Accepts multiword city names and many phrasings.
    """
    q = query_text.strip()
    tail_guard = (
        r'(?=(?:\s*[,;.]?\s*)(?:via|through|stopover(?:\s+in)?|connecting\s+(?:via|through)|stop\s+in|'
        r'on|for|at|tomorrow|today|next|this|return(?:ing)?|coming\s+back|'
        r'leave|leav(?:e|ing)|depart(?:ing)?|after|before|under|with|by)\b|$)'
    )

    # from <A> to <B> [via <C>]
    m = re.search(
        rf'\bfrom\s+([A-Za-z][A-Za-z\s-]{{1,80}}?)\s+to\s+([A-Za-z][A-Za-z\s-]{{1,80}}?){tail_guard}',
        q,
        re.IGNORECASE,
    )
    if m:
        origin = _clean_fragment(m.group(1))
        destination = _clean_fragment(m.group(2))
        via_match = re.search(
            rf'\b(?:via|through|stopover(?:\s+in)?|connecting\s+(?:via|through)|stop\s+in)\s+([A-Za-z][A-Za-z\s-]{{1,80}}?){tail_guard}',
            q,
            re.IGNORECASE,
        )
        via = _clean_fragment(via_match.group(1)) if via_match else None
        return {"origin_text": origin, "destination_text": destination, "via_text": via}

    # <A> to <B> [via <C>]
    m = re.search(
        rf'\b([A-Za-z][A-Za-z\s-]{{1,80}}?)\s+to\s+([A-Za-z][A-Za-z\s-]{{1,80}}?){tail_guard}',
        q,
        re.IGNORECASE,
    )
    if m:
        origin = _clean_fragment(m.group(1))
        destination = _clean_fragment(m.group(2))
        via_match = re.search(
            rf'\b(?:via|through|stopover(?:\s+in)?|connecting\s+(?:via|through)|stop\s+in)\s+([A-Za-z][A-Za-z\s-]{{1,80}}?){tail_guard}',
            q,
            re.IGNORECASE,
        )
        via = _clean_fragment(via_match.group(1)) if via_match else None
        return {"origin_text": origin, "destination_text": destination, "via_text": via}

    # via <C> alone
    m = re.search(
        rf'\b(?:via|through|stopover(?:\s+in)?|connecting\s+(?:via|through)|stop\s+in)\s+([A-Za-z][A-Za-z\s-]{{1,80}}?){tail_guard}',
        q,
        re.IGNORECASE,
    )
    if m:
        return {"origin_text": None, "destination_text": None, "via_text": _clean_fragment(m.group(1))}

    return {"origin_text": None, "destination_text": None, "via_text": None}


def normalize_trip(user_query: str, include_trace: bool = False) -> Dict[str, Any]:
    """
    Build a canonical trip object from raw user_query using regex and centralised resolver.
    Returns dict with keys: origin_iata, destination_iata, via_iata.
    """
    parts = extract_stopover(user_query)

    origin_iata = None
    dest_iata = None
    via_iata = None
    route_trace: Dict[str, Any] = {}

    if parts["origin_text"]:
        origin_iata = resolve_location(parts["origin_text"])
        if origin_iata:
            route_trace["origin_resolved_from"] = parts["origin_text"]

    if parts["destination_text"]:
        dest_iata = resolve_location(parts["destination_text"])
        if dest_iata:
            route_trace["destination_resolved_from"] = parts["destination_text"]

    if parts["via_text"]:
        via_iata = resolve_location(parts["via_text"])
        if via_iata:
            route_trace["via_resolved_from"] = parts["via_text"]

    if include_trace:
        route_trace["raw_fragments"] = {
            "origin_text": parts["origin_text"],
            "destination_text": parts["destination_text"],
            "via_text": parts["via_text"],
        }

    return {
        "origin_iata": origin_iata,
        "destination_iata": dest_iata,
        "via_iata": via_iata,
        "route_trace": route_trace if include_trace else {},
    }


def _infer_route_pair_from_query(user_query: str) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    """
    Infer route pair from compact formats like "DEL BOM", "Delhi Mumbai", etc.
    Returns (origin_iata, dest_iata, trace_dict).
    """
    q = user_query.upper()
    tokens = q.split()

    inferred_origin = None
    inferred_dest = None
    trace: Dict[str, Any] = {}

    if len(tokens) >= 2:
        first, second = tokens[0], tokens[1]
        if is_iata_token(first) and is_iata_token(second):
            inferred_origin = first
            inferred_dest = second
            trace["inferred_from"] = "iata_pair"
        elif is_iata_token(first):
            resolved = resolve_location(tokens[1])
            if resolved:
                inferred_dest = resolved
                inferred_origin = first
                trace["origin_fixed"] = first
                trace["inferred_from"] = "iata_plus_city"
        elif is_iata_token(second):
            resolved = resolve_location(tokens[0])
            if resolved:
                inferred_origin = resolved
                inferred_dest = second
                trace["dest_fixed"] = second
                trace["inferred_from"] = "city_plus_iata"

    return inferred_origin, inferred_dest, trace


def _replace_word_numbers(text: str) -> str:
    """Replace word numbers with digits."""
    normalized = text
    for word, num in WORD_TO_NUM.items():
        normalized = re.sub(rf'\b{word}\b', str(num), normalized)
    return normalized


def _has_explicit_calendar_date(text: str) -> bool:
    """Detect explicit calendar-like dates while excluding duration phrases."""
    month_names = (
        "jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        "jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    )
    return bool(
        re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
        or re.search(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{month_names})(?:\s+\d{{4}})?\b", text)
        or re.search(rf"\b(?:{month_names})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*\d{{4}})?\b", text)
        or re.search(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})\b", text)
    )


def _has_ambiguous_calendar_without_year(text: str) -> bool:
    """Detect month/day style dates without an explicit year."""
    month_names = (
        "jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        "jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    )
    return bool(
        re.search(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{month_names})\b", text)
        and not re.search(rf"\b(?:{month_names})\s+\d{{4}}\b", text)
    )


def _expand_relative_date_refs(text: str, base_date: Any) -> str:
    """Expand relative date references like 'next Friday' to concrete dates."""
    from datetime import timedelta
    day_name_to_delta = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }
    q = text.lower()
    for day_name, delta in day_name_to_delta.items():
        pattern = rf"\bnext\s+{day_name}\b"
        if re.search(pattern, q):
            days_ahead = (delta - base_date.weekday() + 7) % 7
            if days_ahead == 0:
                days_ahead = 7
            target = base_date + timedelta(days=days_ahead)
            q = re.sub(pattern, target.strftime("%Y-%m-%d"), q)
    return q