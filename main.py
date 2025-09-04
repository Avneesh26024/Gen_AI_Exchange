from Verifier_Agent import verifier_tool
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage
from typing import TypedDict, List
from Retriever.Retriever_Agent import retriever_agent
from Tools.Human_Response import human_response
from prompts import main_prompt
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from vertexai import agent_engines
from langchain.schema import AIMessage
from Verifier_Agent import extract_final_ai_message, verifier_tool

# ---- Model + Safety settings ----
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
tools = [retriever_agent, verifier_tool, human_response]
agent = agent_engines.LanggraphAgent(
    model="gemini-2.5-flash-lite",  # Pass the model name as a string
    model_kwargs=model_kwargs,
    tools=tools
)
# The AgentState should expect a list of BaseMessage objects, not dicts
class AgentState(TypedDict):
    messages: List
def response(state: AgentState) -> AgentState:
    """
    Calls the ReAct agent to decide the next step or generate a final response.
    The agent's output *is* the new state.
    """
    print("---CALLING AGENT---")
    last_human_message = state["messages"][-1]

    # Construct the message list in the correct format
    messages_for_agent = {
        "messages": [
            {"type": "system", "content": main_prompt(state)},
            # CORRECTED LINE: Use .content to get the string
            {"type": "human", "content": last_human_message.content}
        ]
    }
    ans = agent.query(input=messages_for_agent)
    final_answer = extract_final_ai_message(ans)

    state["messages"].append(
        AIMessage(content=final_answer if final_answer else "I'm sorry, I couldn't generate a response."))
    return state

graph = StateGraph(AgentState)
graph.add_node("response", response)
graph.set_entry_point("response")
graph.set_finish_point("response")
classifier = graph.compile()

# --- MAJOR CHANGE IN THE CHAT LOOP ---
if __name__ == "__main__":
    # State is now managed by a single dictionary that gets updated each turn.
    # conversation_state: AgentState = {
    #     "messages": []
    # }
    #
    # print("Multiturn chatbot started. Type 'exit' or 'quit' to stop.")
    # try:
    #     while True:
    #         user_input = input("You: ").strip()
    #         if not user_input:
    #             continue
    #         if user_input.lower() in ("exit", "quit", "q"):
    #             print("Goodbye.")
    #             break
    #
    #         # Append user message as a HumanMessage object for LangChain compatibility
    #         conversation_state["messages"].append(HumanMessage(content=user_input))
    #
    #         # Invoke the graph with the current state
    #         final_state = classifier.invoke(conversation_state)
    #
    #         # The last message object from the agent's output has the reply
    #         last_message = final_state["messages"][-1]
    #         print(f"Bot: {last_message.content}")
    #
    #         # **CRITICAL STEP**: Update the state for the next turn to maintain memory
    #         conversation_state = final_state
    #
    # except KeyboardInterrupt:
    #     print("\nInterrupted. Exiting.")

    messages_for_agent = {
        "messages": [
            {"type": "human", "content": "eiffel tower is in india"}
        ]
    }
     # 2. Pass the correctly formatted dictionary to the agent
    ans = agent.query(input=messages_for_agent)

    print(ans)