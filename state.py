from typing import TypedDict, List, Optional, Literal
from langchain_core.messages import BaseMessage

class MyForgeState(TypedDict):
    messages: List[BaseMessage]
    conversation_history: List[str]
    problem_description: str
    current_diagnosis: Optional[str]
    is_load_bearing_risk: Optional[bool]
    last_response: Optional[str]
    awaiting_user_confirmation: bool
    mode: Literal["human", "robot"]
    conversation_complete: bool