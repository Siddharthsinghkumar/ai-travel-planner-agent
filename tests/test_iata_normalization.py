#tests/test_iata_normalization.py
import pytest
from agents.planner_agent import normalize_airport

def test_explicit_iata_uppercase():
    assert normalize_airport("DEL") == "DEL"

def test_explicit_iata_lowercase():
    assert normalize_airport("bom") == "BOM"

def test_non_iata_city():
    result = normalize_airport("delhi")
    assert result is not None
    assert len(result) == 3

from agents.planner_agent import parse_intent

def test_parse_intent_iata_codes():
    intent = parse_intent("DEL to BOM tomorrow")
    assert intent.origin_iata == "DEL"
    assert intent.destination_iata == "BOM"

import asyncio
from agents.planner_agent import _plan_trip_internal

@pytest.mark.asyncio
async def test_origin_override():
    result = await _plan_trip_internal(
        origin="DEL",
        destination="BOM",
        user_query="random text",
        skip_llm=True
    )

    assert result.best_flight is not None