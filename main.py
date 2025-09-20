from Verifier_Agent import verifier_tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory, ChatVertexAI
from prompts import main_prompt
from typing import TypedDict, List
from Retriever.Retriever_Agent import retriever_agent
from Tools.Human_Response import human_response
from langchain.prompts import ChatPromptTemplate


###history
###retriever


# --- Model Configuration (Unchanged) ---
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
model = ChatVertexAI(
    model_name="gemini-2.5-flash-lite",
    **model_kwargs
)


# --- Agent State Definition (Unchanged) ---
class AgentState(TypedDict):
    messages: List[BaseMessage]
    verified_results: str
    relevant_context: str
    condensed_query: str
    image_path: List[str]


# --- All Node Definitions (Final Version) ---

def router_entry_node(state: AgentState) -> AgentState:
    """A simple node that acts as the starting point for the routing logic."""
    print("---ENTERING GRAPH, PREPARING TO ROUTE---")
    return state


def condense_query(state: AgentState) -> AgentState:
    """Condenses the chat history and latest user query into a standalone question."""
    print("---NODE: CONDENSING QUERY---")
    user_message = state["messages"][-1].content
    history = state["messages"][:-1]
    if not history:
        state["condensed_query"] = user_message
        print(f"---CONDENSED QUERY (No history): {user_message}---")
        return state

    formatted_history = "\n".join(
        [f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}" for msg in history])
    condensing_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question, in its original language. Do not answer the question, just reformulate it."),
        ("human", "Chat History:\n{chat_history}\n\nFollow Up Input: {question}"),
    ])
    condenser_chain = condensing_prompt | model
    response = condenser_chain.invoke({"chat_history": formatted_history, "question": user_message})
    state["condensed_query"] = response.content
    print(f"---CONDENSED QUERY: {response.content}---")
    return state


# MODIFIED: Upgraded to a 3-way router for conversational flexibility
def decide_route(state: AgentState) -> str:
    """The initial router, now with a third 'general_conversation' option."""
    print("---ROUTER: DECIDING INITIAL ROUTE---")
    query = state["condensed_query"]
    router_prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are an expert at routing user queries. Classify the query into one of three categories:

         1. 'answer_from_context': The user is asking a clear question about a topic that can likely be answered by retrieving existing, verified information from a database.
         2. 'verify_new_claim': The user is presenting a new, specific factual statement, URL, or piece of information that needs to be fact-checked from scratch using web searches.
         3. 'general_conversation': The user is asking for an opinion, an elaboration on a previous point, a subjective question, or is making a simple conversational remark (e.g., "what?", "thanks", "tell me more", "why is that?").

         Respond with *only* the category name."""),
        ("human", "{user_query}"),
    ])
    router_chain = router_prompt | model
    route = router_chain.invoke({"user_query": query})
    route_content = route.content.strip().lower()

    if "verify_new_claim" in route_content:
        print("---ROUTE: To VERIFY_CLAIM---")
        return "verify_claim"
    elif "answer_from_context" in route_content:
        print("---ROUTE: To ANSWER_FROM_CONTEXT---")
        return "answer_from_context"
    else:
        print("---ROUTE: To GENERAL_CONVERSATION---")
        return "general_conversation"


def retrieve_context(state: AgentState) -> AgentState:
    """Node to retrieve context using the condensed query."""
    print("---NODE: RETRIEVE CONTEXT---")
    query = state["condensed_query"]
    context_result = retriever_agent(query)
    chunks = context_result.get("retrieved_chunks", [])
    context_texts = [chunk.get("text", "") for chunk in chunks]
    state["relevant_context"] = "\n\n".join(context_texts)
    print("---CONTEXT RETRIEVED---")
    return state


def check_relevance(state: AgentState) -> str:
    """The relevance-checking router, using the condensed query."""
    print("---ROUTER: CHECKING RELEVANCE---")
    query = state["condensed_query"]
    context = state["relevant_context"]
    if not context.strip():
        print("---ROUTE: No context found, must verify.---")
        return "verify_claim"

    relevance_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a relevance-checking AI... Respond with only 'yes' or 'no'."),
        ("human", "CONTEXT:\n{context}\n\nQUERY:\n{query}"),
    ])
    relevance_chain = relevance_prompt | model
    response = relevance_chain.invoke({"context": context, "query": query})
    if "yes" in response.content.strip().lower():
        print("---ROUTE: Context is relevant. Proceeding to answer.---")
        return "synthesize_answer"
    else:
        print("---ROUTE: Context is NOT relevant. Rerouting to verification.---")
        return "verify_claim"


def synthesize_answer(state: AgentState) -> AgentState:
    """Node to synthesize a final answer using the condensed query."""
    print("---NODE: SYNTHESIZE ANSWER---")
    query = state["condensed_query"]
    context = state["relevant_context"]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant... Answer the user's question based *only* on the context."),
        ("human", "Context:\n{context}\n\nUser Question:\n{query}"),
    ])
    answer_chain = prompt | model
    response_msg = answer_chain.invoke({"context": context, "query": query})
    state["messages"].append(response_msg)
    return state


# NEW NODE: Handles conversational turns that don't need tools
def handle_conversation(state: AgentState) -> AgentState:
    """Node to handle general conversation without using heavy tools."""
    print("---NODE: HANDLE CONVERSATION---")
    condensed_query = state["condensed_query"]
    messages_str = "\n".join([f"{msg.type}: {msg.content}" for msg in state["messages"]])

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful and conversational AI assistant. Answer the user's question based on the chat history."),
        ("human",
         "Here is the conversation history:\n{messages_str}\n\nBased on this, please answer the following question: {query}")
    ])
    chain = prompt | model
    response_msg = chain.invoke({"messages_str": messages_str, "query": condensed_query})
    state["messages"].append(response_msg)
    return state


# MODIFIED: Fixed to use condensed_query to prevent context loss
def verify_and_respond(state: AgentState) -> AgentState:
    """Node for ReAct verification, now using the condensed query."""
    print("---NODE: VERIFY NEW CLAIM (ReAct Agent)---")
    react_agent_executor = create_react_agent(model=model, tools=[verifier_tool, human_response], prompt=main_prompt())

    # CRITICAL FIX: Replace the last user message with the condensed query
    # to give the ReAct agent the full, unambiguous context.
    messages_for_agent = state["messages"][:-1] + [HumanMessage(content=state["condensed_query"])]

    agent_output = react_agent_executor.invoke({"messages": messages_for_agent})

    all_agent_messages = agent_output.get("messages", [])
    verification_results = [str(msg.content) for msg in all_agent_messages if isinstance(msg, ToolMessage)]
    if verification_results:
        existing_results = state.get("verified_results", "")
        new_results_str = "\n".join(verification_results)
        state[
            "verified_results"] = f"{existing_results}\n---\n{new_results_str}" if existing_results else new_results_str
        print(f"--- 📝 VERIFICATION RESULT SAVED TO STATE ---")

    final_ai_message = all_agent_messages[-1]
    state["messages"].append(final_ai_message)
    return state


# --- 4. Graph Definition (Final Version) ---
graph = StateGraph(AgentState)

graph.add_node("router", router_entry_node)
graph.add_node("condense_query", condense_query)
graph.add_node("retrieve_context", retrieve_context)
graph.add_node("synthesize_answer", synthesize_answer)
graph.add_node("verify_claim", verify_and_respond)
graph.add_node("handle_conversation", handle_conversation) 

graph.set_entry_point("router")
graph.add_edge("router", "condense_query")

graph.add_conditional_edges(
    "condense_query",
    decide_route,
    {
        "answer_from_context": "retrieve_context",
        "verify_claim": "verify_claim",
        "general_conversation": "handle_conversation",
    },
)
graph.add_conditional_edges(
    "retrieve_context",
    check_relevance,
    {
        "synthesize_answer": "synthesize_answer",
        "verify_claim": "verify_claim",
    },
)

graph.add_edge("synthesize_answer", END)
graph.add_edge("verify_claim", END)
graph.add_edge("handle_conversation", END)  # Added new end point

<<<<<<< HEAD
agent = graph.compile()
=======
agent = graph.compile(checkpointer=memory)





>>>>>>> ac7e9d7 (Sync local files)
agent.get_graph().print_ascii()

# --- 5. Main Chat Loop (Unchanged) ---
if __name__ == "__main__":
    initial_state: AgentState = {"messages": [], "verified_results": "", "relevant_context": "", "condensed_query": ""}
    print("✅ Misinfo Classifier Agent started. Type 'exit' or 'quit' to stop.")
    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input or user_input.lower() in ("exit", "quit", "q"): break

            current_messages = initial_state.get("messages", [])
            current_messages.append(HumanMessage(content=user_input))

            agent_input = {"messages": current_messages}
            final_state = agent.invoke(agent_input)

<<<<<<< HEAD
            last_message = final_state["messages"][-1]
            print(f"Bot: {last_message.content}")
            initial_state = final_state
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
=======
#             last_message = final_state["messages"][-1]
#             print(f"Bot: {last_message.content}")
#             initial_state = final_state
#     except KeyboardInterrupt:
#         print("\nInterrupted. Exiting.")
#     except Exception as e:
#         print(f"\nAn error occurred: {e}")



>>>>>>> ac7e9d7 (Sync local files)
