from langgraph.graph import StateGraph, END
from state import MyForgeState
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from retriever import get_myforge_retriever

llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.2)

def create_myforge_graph():
    workflow = StateGraph(MyForgeState)

    def chat_node(state: MyForgeState):
        retriever = get_myforge_retriever()

        if retriever:
            docs = retriever.invoke(state.get("problem_description", ""))
            context = "\n\n".join([d.page_content for d in docs])
        else:
            context = "No additional knowledge available."

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""You are MyForge, a helpful home maintenance assistant.
Use the following knowledge when relevant:

{context}"""),
            HumanMessage(content=state.get("problem_description", ""))
        ])
        response = llm.invoke(prompt.format_messages())
        return {"last_response": response.content}

    workflow.add_node("chat", chat_node)
    workflow.set_entry_point("chat")
    workflow.add_edge("chat", END)
    return workflow.compile()
