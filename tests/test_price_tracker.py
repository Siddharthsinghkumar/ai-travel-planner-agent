#tests/test_price_tracker.py
import pytest
from tools.price_tracker import parse_price_insights, format_price_insights_for_llm

def test_price_insights_parsing_and_formatting():
    # 1. Mock the SerpApi JSON response block
    raw_serpapi_response = {
        "price_insights": {
            "price_level": "low",
            "typical_price_range": [5500, 7200],
            "lowest_price": 4200,
            "price_history": [[1670000000, 6000], [1670086400, 4200]]  # Simulates a falling trend
        }
    }
    
    # 2. Test Parsing
    insights = parse_price_insights(raw_serpapi_response)
    
    assert insights is not None
    assert insights.price_level == "low"
    assert insights.typical_low_inr == 5500.0
    assert insights.typical_high_inr == 7200.0
    assert insights.current_price_inr == 4200.0
    assert insights.trend == "falling"
    
    # 3. Test Formatting for the LLM
    formatted_string = format_price_insights_for_llm(insights)
    
    # Assert the required keywords made it into the LLM prompt string
    assert "LOW vs. typical" in formatted_string
    assert "₹5,500–₹7,200" in formatted_string
    assert "falling" in formatted_string
    assert "Recommend booking soon" in formatted_string