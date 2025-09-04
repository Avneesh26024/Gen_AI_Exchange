from Verifier_Agent import verifier_tool
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory, ChatVertexAI
from prompts import main_prompt
from typing import TypedDict, List
from Retriever.Retriever_Agent import retriever_agent
from Tools.Human_Response import human_response

safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}
model_kwargs = {
    "temperature": 0.28,
    "max_output_tokens": 1000,
    "top_p": 0.95,
    "top_k": None,
    "safety_settings": safety_settings,
}

# The summarizer_llm and AgentState remain unchanged
model = ChatVertexAI(
    model_name="gemini-2.5-flash-lite", # Using a stable model name
    **model_kwargs
)

# The AgentState should expect a list of BaseMessage objects, not dicts
class AgentState(TypedDict):
    messages: List[str]
    verified_results: str
    relevant_context: str

# This node remains largely the same
def attach_context(state: AgentState) -> AgentState:
    """
    Attach relevant context to the agent state.
    """
    print("---ATTACHING CONTEXT---")
    # Get the last message object
    last_msg = state["messages"][-1]
    # The query is in its .content attribute
    query = last_msg.content

    try:
        if hasattr(retriever_agent, "invoke"):
            context_result = retriever_agent.invoke(query)
        else:
            context_result = retriever_agent(query)
    except Exception as e:
        print(f"Retriever tool error: {e}")
        context_result = {}

    if context_result and "retrieved_chunks" in context_result:
        chunks = context_result.get("retrieved_chunks", [])
        context_texts = [chunk.get("text", "") for chunk in chunks]
        state["relevant_context"] = "\n".join(context_texts)
    else:
        state["relevant_context"] = ""

    return state

# --- MAJOR CHANGE HERE ---
# This function is now much simpler.
def response(state: AgentState) -> AgentState:
    """
    Calls the ReAct agent to decide the next step or generate a final response.
    The agent's output *is* the new state.
    """
    print("---CALLING REACT AGENT---")
    agent = create_react_agent(
        model=model,
        tools=[
            human_response,
            retriever_agent,
            verifier_tool
        ],
        # The prompt is now passed directly, not wrapped in a function call
        prompt=main_prompt(state)
    )
    # The agent executor directly returns the final state with the updated messages.
    # No need to manually append messages here.
    final_state = agent.invoke(state)
    return final_state


graph = StateGraph(AgentState)
graph.add_node("attach_context", attach_context)
graph.add_node("response", response)
graph.set_entry_point("attach_context")
graph.add_edge("attach_context", "response")
graph.set_finish_point("response")
agent = graph.compile()

# --- MAJOR CHANGE IN THE CHAT LOOP ---
if __name__ == "__main__":
    # State is now managed by a single dictionary that gets updated each turn.
    conversation_state: AgentState = {
        "messages": [],
    }

    print("Multiturn chatbot started. Type 'exit' or 'quit' to stop.")
    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye.")
                break

            # Append user message as a HumanMessage object for LangChain compatibility
            conversation_state["messages"].append(HumanMessage(content=user_input))

            # Invoke the graph with the current state
            final_state = agent.invoke(conversation_state)

            # The last message object from the agent's output has the reply
            last_message = final_state["messages"][-1]
            print(f"Bot: {last_message.content}")

            # **CRITICAL STEP**: Update the state for the next turn to maintain memory
            conversation_state = final_state

    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")