from langgraph.graph import StateGraph, END
from state import MyForgeState
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from retriever import get_myforge_retriever

retriever = get_myforge_retriever()

def create_myforge_graph():
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.2)

    workflow = StateGraph(MyForgeState)

    def chat_node(state: MyForgeState):
        history = "\n".join(state.get("conversation_history", []))
        docs = retriever.invoke(state.get("problem_description", ""))
        context = "\n\n".join([d.page_content for d in docs])

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""You are MyForge, a helpful and patient home maintenance guide.
Continue the conversation naturally. Ask clarifying questions when needed.
Use this knowledge when relevant:

{context}"""),
            HumanMessage(content=f"Conversation so far:\n{history}\n\nUser: {state.get('problem_description')}")
        ])
        response = llm.invoke(prompt.format_messages())
        return {
            "last_response": response.content,
            "conversation_history": state.get("conversation_history", []) + [f"User: {state.get('problem_description')}", f"MyForge: {response.content}"]
        }

    workflow.add_node("chat", chat_node)
    workflow.set_entry_point("chat")
    workflow.add_edge("chat", END)
    return workflow.compile()