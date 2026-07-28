from typing import TypedDict, Annotated, Optional
import operator

def merge_list(a: list, b: list) -> list:
    return a + b

class TravelPlannerState(TypedDict):
    """LangGraph state for the V2 travel planner."""
    thread_id: str
    user_query: str
    session_id: str
    
    # NLP extraction
    parsed_intent: Optional[dict]
    
    # Tool outputs
    flights: Annotated[list[dict], merge_list]
    weather: Annotated[list[dict], merge_list]
    
    # Graph execution state
    current_step: str
    errors: Annotated[list[str], merge_list]
    
    # Final output
    final_plan: Optional[str]
    booking_urls: Annotated[list[str], merge_list]
