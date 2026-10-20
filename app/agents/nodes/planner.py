from app.agents.state import AgentState 
#from langchain_groq import ChatGroq 
from app.gateway.client import get_langchain_llm 
from app.config import settings 
import logfire 

llm = get_langchain_llm(feature="planner")


def planner_node(state:AgentState):
    """
    The Planner determines if a search is needed based on the ENTIRE conversation """

    #Get the conversation hostory (excluding the latest message )
    history= ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    user_message = state["messages"][-1]["content"] if state['messages'] else ""

    prompt = f"""
You are an intelligent Assistant Planner. 
Analyze the conversation history and the latest user message 

CONVERSATION HISTORY : 
{history}

LATEST MESSAGE:
"{user_message}

Task : 
1. If the latest message is a greeting (hi,hello) or a question that can be answered using ONLY the conversation history above (e.g, "What is my name"), respond with 'CONVERSATIONAL'.
2. If it is a technical question about Kubernetes, Intel, or Networking that requires fresh documentation, output a refined search query .

Output ONLY the word 'CONVERSATIONAL' or the search query. Do not explain .
"""
    with logfire.span("🧠 Planner Decision"):
          decision = llm.invoke(prompt).content.strip()
          logfire.info("Planner",user_message=user_message, decision=decision)
    if decision == "CONVERSATIONAL":
         return {
              "current_query":"CONVERSATIONAL",
              "status":"Handling concversationally (using memory)...",
              "plan":["Intent: Conversational/Memory", "Retrieval : Skipped"]
         }
    return {
         "current_query":decision,
         "status":f"Technical research needed . Searching for : {decision}",
         "plan":["Intent: Technical", f"search Term: {decision}"]
    }