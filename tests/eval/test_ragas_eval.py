import pytest
from tests.eval.eval_harness import EvaluationHarness

def test_merlin_judge_evaluates_faithfulness_and_relevancy():
    harness = EvaluationHarness(judge_type="merlin")
    # Mocked LangGraph trajectory
    mock_trajectory = {
        "query": "I want to fly to Paris next week",
        "state_history": ["Fetched flights: Air France 123", "Fetched flights: Lufthansa 456"],
        "final_response": "You can fly to Paris on Air France 123 or Lufthansa 456."
    }
    
    results = harness.evaluate_trajectory(mock_trajectory)
    
    assert "faithfulness" in results
    assert "relevancy" in results
    assert results["faithfulness"] == 0.95
    assert results["relevancy"] == 0.90

def test_local_stack_judge_evaluates_faithfulness_and_relevancy():
    harness = EvaluationHarness(judge_type="local")
    # Mocked LangGraph trajectory
    mock_trajectory = {
        "query": "Find me a hotel in Tokyo",
        "state_history": ["Fetched hotels: Shinjuku Prince", "Fetched hotels: Tokyo Hilton"],
        "final_response": "You can stay at the Shinjuku Prince or Tokyo Hilton."
    }
    
    results = harness.evaluate_trajectory(mock_trajectory)
    
    assert "faithfulness" in results
    assert "relevancy" in results
    assert results["faithfulness"] == 0.85
    assert results["relevancy"] == 0.80

def test_evaluation_harness_invalid_judge():
    with pytest.raises(ValueError):
        EvaluationHarness(judge_type="invalid_judge")
