from Verifier_Agent import verifier_tool
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from vertexai.preview.generative_models import GenerativeModel
from typing import TypedDict, List
from Retriever.Retriever_Agent import retriever_agent
from Tools.Human_Response import human_response
from prompts import main_prompt




model = GenerativeModel("gemini-2.5-flash-lite")

class AgentState(TypedDict):
    id: int
    messages: List[str]
    verified_results: str
    relevant_context: str

def attach_context(state: AgentState)-> AgentState:
    """
    Attach relevant context to the agent state.
    :param state:
    :return:
    """
    context = retriever_agent.send_message(state["messages"][-1])
    state["relevant_context"] = context.text
    return state
def response(state: AgentState)-> AgentState:
    """
    Gives final response to users.
    :param state:
    :return:
    """
    agent = create_react_agent(
        model = model,
        tools=[
            verifier_tool,
            human_response,
            retriever_agent
        ],
        messages_modifier=main_prompt()
    )
    result = agent.invoke(state)
    state["messages"] = result["messages"]
    return state
