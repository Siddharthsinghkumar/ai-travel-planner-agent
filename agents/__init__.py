import sys
import os

# Add the parent directory of the current file (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './home/sidd/project/llm-travel-agent.')))

from agents.planner_agent import plan_trip

# ... rest of your imports and code

