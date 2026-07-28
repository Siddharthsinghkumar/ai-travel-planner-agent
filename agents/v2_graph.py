import logging
from langgraph.graph import StateGraph, START, END
from agents.v2_state import TravelPlannerState
from langgraph.checkpoint.memory import MemorySaver


logger = logging.getLogger(__name__)

async def parse_intent_node(state: TravelPlannerState) -> dict:
    logger.info("parse_intent_node running", extra={"thread_id": state.get("thread_id")})
    # Placeholder: In full implementation, call extract_intent_local
    # For now, we mock an intent
    return {
        "current_step": "parse_intent", 
        "parsed_intent": {"type": "flight_search", "origin": "NYC", "dest": "PAR"}
    }

def route_tools(state: TravelPlannerState) -> list[str]:
    """Conditional edge to route to tools based on intent."""
    intent = state.get("parsed_intent", {})
    routes = []
    if intent.get("type") == "flight_search":
        routes.append("fetch_flights")
        routes.append("fetch_weather")
    else:
        routes.append("generate_plan")
    return routes

async def fetch_flights_node(state: TravelPlannerState) -> dict:
    logger.info("fetch_flights_node running")
    # Placeholder for actual flight API call
    return {"flights": [{"airline": "Air France", "price": 500}], "current_step": "fetch_flights"}

async def fetch_weather_node(state: TravelPlannerState) -> dict:
    logger.info("fetch_weather_node running")
    # Placeholder for actual weather API call
    return {"weather": [{"temp": 72, "desc": "Sunny"}], "current_step": "fetch_weather"}

async def generate_plan_node(state: TravelPlannerState) -> dict:
    logger.info("generate_plan_node running")
    # Placeholder for actual LLM generation
    plan = "Based on your request, here is your flight to PAR with sunny weather."
    return {"final_plan": plan, "current_step": "generate_plan"}

workflow = StateGraph(TravelPlannerState)
workflow.add_node("parse_intent", parse_intent_node)
workflow.add_node("fetch_flights", fetch_flights_node)
workflow.add_node("fetch_weather", fetch_weather_node)
workflow.add_node("generate_plan", generate_plan_node)

workflow.add_edge(START, "parse_intent")
workflow.add_conditional_edges("parse_intent", route_tools, ["fetch_flights", "fetch_weather", "generate_plan"])

# Fan-in from tools
workflow.add_edge("fetch_flights", "generate_plan")
workflow.add_edge("fetch_weather", "generate_plan")

workflow.add_edge("generate_plan", END)

# In-memory checkpointer for now until Section 5 (Infra agent) finishes Postgres.
memory = MemorySaver()

# Compile the graph
v2_agent = workflow.compile(checkpointer=memory)
