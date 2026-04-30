"""Helpers for stable LLM explanation validation.

These rules are intentionally conservative:
- Prefer structured truth over phrasing style.
- Fail only on detectable semantic contradictions.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_NEGATION_MARKERS = (
    "no ",
    "not ",
    "unavailable",
    "could not",
    "couldn't",
    "unable to",
    "not available",
    "no exact",
    "closest alternative",
    "closest option",
    "not found",
)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text or "") if s and s.strip()]


def _contains_any(text: str, needles: Iterable[str]) -> bool:
    t = (text or "").lower()
    return any(n in t for n in needles)


def _looks_negated(sentence: str) -> bool:
    return _contains_any(sentence, _NEGATION_MARKERS)


def _airline_matches_preference(selected_airline: str, preferred_airlines: Iterable[str]) -> bool:
    selected = (selected_airline or "").strip().lower()
    if not selected:
        return False
    for pref in preferred_airlines or ():
        p = (pref or "").strip().lower()
        if not p:
            continue
        if p in selected or selected in p:
            return True
    return False


def detect_relaxed_preferred_airline_contradiction(
    llm_text: str,
    preferred_airlines: Iterable[str],
    selected_airline: str,
) -> Optional[str]:
    """Return failure reason if LLM claims preferred-airline satisfaction when relaxed."""
    prefs = [p.strip().lower() for p in (preferred_airlines or []) if (p or "").strip()]
    if not prefs:
        return None
    if _airline_matches_preference(selected_airline, prefs):
        return None

    positive_markers = (
        "best",
        "recommended",
        "selected",
        "matches your preference",
        "meets your preference",
        "satisfies your preference",
    )

    for sentence in _split_sentences(llm_text):
        s = sentence.lower()
        if not any(pref in s for pref in prefs):
            continue
        if _looks_negated(s):
            continue
        if _contains_any(s, positive_markers):
            return (
                "LLM claims preferred-airline match despite relaxed preferred airline "
                f"(selected airline='{selected_airline}')"
            )
        # Also fail on explicit "<preferred> flight is the best/recommended" patterns.
        for pref in prefs:
            if re.search(rf"\b{re.escape(pref)}\b.*\b(flight|airline)\b", s) and _contains_any(
                s, ("is the best", "best option", "recommended option")
            ):
                return (
                    "LLM claims preferred-airline recommendation despite relaxed preferred airline "
                    f"(selected airline='{selected_airline}')"
                )
    return None


def detect_layover_contradiction(
    llm_text: str,
    best_flight: dict,
    layover_limit_minutes: Optional[int],
) -> Optional[str]:
    """Return failure reason only for semantic layover contradictions."""
    if not layover_limit_minutes:
        return None

    text = llm_text or ""
    stops = (best_flight or {}).get("stops")
    layovers = (best_flight or {}).get("layover_durations_min") or []
    try:
        layovers = [int(x) for x in layovers if x is not None]
    except Exception:
        layovers = []

    # If non-stop is selected, do not fail on awkward constraint restatements.
    # Fail only on explicit positive layover claims for the selected flight.
    if str(stops) == "0":
        explicit_positive_layover_patterns = (
            r"\b(has|with|includes|involves)\b.{0,25}\blayover\b",
            r"\blayover of\b",
            r"\blayover\b.{0,20}\b\d+\s*(min|mins|minute|minutes|hour|hours|hr|hrs)\b",
            r"\b\d+\s*(min|mins|minute|minutes|hour|hours|hr|hrs)\b.{0,20}\blayover\b",
            r"\bconnecting time\b.{0,20}\b\d+\b",
        )
        exemptions = (
            "non-stop",
            "nonstop",
            "no layover",
            "zero layover",
            "without layover",
            "requirement",
            "preference",
            "max layover",
            "under your limit",
            "less than",
            "under",
            "at most",
        )
        for sentence in _split_sentences(text):
            s = sentence.lower()
            if _looks_negated(s):
                continue
            if any(re.search(p, s) for p in explicit_positive_layover_patterns):
                if not _contains_any(s, exemptions):
                    return "LLM implies non-stop flight has an actual layover"

    # If structured data violates the layover limit, explanation must not claim compliance.
    if layovers and max(layovers) > int(layover_limit_minutes):
        compliance_claims = (
            r"\b(meets|satisfies|within)\b.{0,30}\b(max\s+)?layover\b",
            r"\blayover\b.{0,20}\b(less than|under|within|at most)\b",
        )
        for sentence in _split_sentences(text):
            s = sentence.lower()
            if _looks_negated(s):
                continue
            if any(re.search(p, s) for p in compliance_claims):
                return (
                    "LLM claims layover-limit compliance but structured layover durations "
                    "exceed the limit"
                )

    # If structured data has connections, explanation must not call it non-stop/direct.
    try:
        stops_int = int(stops)
    except Exception:
        stops_int = None
    if stops_int is not None and stops_int > 0:
        for sentence in _split_sentences(text):
            s = sentence.lower()
            if _looks_negated(s):
                continue
            if _contains_any(s, ("non-stop", "nonstop", "direct flight", "no stops")):
                return "LLM claims non-stop/direct but selected flight has stops"

    return None

