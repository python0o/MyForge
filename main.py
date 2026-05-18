from dotenv import load_dotenv
from graph import create_myforge_graph
from state import MyForgeState

load_dotenv()
graph = create_myforge_graph()

state: MyForgeState = {
    "messages": [],
    "problem_description": input("Describe the problem: "),
    "photo_description": None,
    "diagnosis": None,
    "is_load_bearing_risk": None,
    "current_step": 0,
    "guidance_history": [],
    "awaiting_confirmation": False,
    "last_proposed_step": None,
    "mode": "human",
    "final_output": None,
    "conversation_complete": False
}

result = graph.invoke(state)
print(result.get("final_output"))