from Verifier_Agent import verifier_tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage, AIMessage
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory, ChatVertexAI
from prompts import main_prompt
from langgraph.checkpoint.memory import MemorySaver

from typing import TypedDict, List
from Retriever.Retriever_Agent import retriever_agent
from Tools.Human_Response import human_response
from langchain.prompts import ChatPromptTemplate

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


# --- Agent State Definition (Fixed) ---
class AgentState(TypedDict):
    messages: List[BaseMessage]
    verified_results: str
    relevant_context: str
    condensed_query: str
    image_path: List[str]


# --- Helper Functions ---
def get_last_human_message(messages: List[BaseMessage]) -> str:
    """Extract the last human message content."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content
    return ""

def get_chat_history(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Get all messages except the last human message."""
    if not messages:
        return []
    # Find the last human message index
    last_human_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_human_idx = i
            break
    
    return messages[:last_human_idx] if last_human_idx > 0 else []


# --- All Node Definitions (Fixed Version) ---

def router_entry_node(state: AgentState) -> AgentState:
    """A simple node that acts as the starting point for the routing logic."""
    print("---ENTERING GRAPH, PREPARING TO ROUTE---")
    return state


def condense_query(state: AgentState) -> AgentState:
    """Condenses the chat history and latest user query into a standalone question."""
    print("---NODE: CONDENSING QUERY---")
    
    # Get the latest user message and chat history
    user_message = get_last_human_message(state["messages"])
    history = get_chat_history(state["messages"])
    
    if not history or len(history) == 0:
        state["condensed_query"] = user_message
        print(f"---CONDENSED QUERY (No history): {user_message}---")
        return state

    # Format history properly
    formatted_history = "\n".join([
        f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}" 
        for msg in history[-10:]  # Keep last 10 messages to avoid token limit
    ])
    
    condensing_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question, in its original language. Do not answer the question, just reformulate it."),
        ("human", "Chat History:\n{chat_history}\n\nFollow Up Input: {question}"),
    ])
    
    condenser_chain = condensing_prompt | model
    response = condenser_chain.invoke({
        "chat_history": formatted_history, 
        "question": user_message
    })
    
    state["condensed_query"] = response.content
    print(f"---CONDENSED QUERY: {response.content}---")
    return state


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
    
    try:
        context_result = retriever_agent(query)
        chunks = context_result.get("retrieved_chunks", [])
        context_texts = [chunk.get("text", "") for chunk in chunks if chunk.get("text", "").strip()]
        state["relevant_context"] = "\n\n".join(context_texts)
        print(f"---CONTEXT RETRIEVED: {len(context_texts)} chunks---")
    except Exception as e:
        print(f"---ERROR RETRIEVING CONTEXT: {e}---")
        state["relevant_context"] = ""
    
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
        ("system", 
         """You are a relevance-checking AI. Determine if the provided context contains information that can answer the user's query.
         
         Respond with only 'yes' if the context contains relevant information to answer the query.
         Respond with only 'no' if the context does not contain relevant information."""),
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
        ("system", 
         """You are a helpful assistant. Answer the user's question based on the provided context.
         If the context doesn't contain enough information to fully answer the question, say so clearly.
         Be concise and accurate."""),
        ("human", "Context:\n{context}\n\nUser Question:\n{query}"),
    ])
    
    answer_chain = prompt | model
    response = answer_chain.invoke({"context": context, "query": query})
    
    # Create AIMessage and add to state
    ai_message = AIMessage(content=response.content)
    state["messages"].append(ai_message)
    
    return state


def handle_conversation(state: AgentState) -> AgentState:
    """Node to handle general conversation without using heavy tools."""
    print("---NODE: HANDLE CONVERSATION---")
    
    # Use the full message history for conversational context
    messages = state["messages"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are a helpful and conversational AI assistant. Respond naturally to the user's message 
         based on the conversation history. Be friendly, concise, and helpful."""),
        ("human", "Please respond to the latest message in this conversation: {query}")
    ])
    
    chain = prompt | model
    response = chain.invoke({"query": state["condensed_query"]})
    
    # Create AIMessage and add to state
    ai_message = AIMessage(content=response.content)
    state["messages"].append(ai_message)
    
    return state


def verify_and_respond(state: AgentState) -> AgentState:
    """Node for ReAct verification, now using the condensed query."""
    print("---NODE: VERIFY NEW CLAIM (ReAct Agent)---")
    
    try:
        react_agent_executor = create_react_agent(
            model=model, 
            tools=[verifier_tool, human_response], 
            prompt=main_prompt()
        )

        # Use condensed query for verification but preserve message history
        messages_for_agent = state["messages"][:-1] + [HumanMessage(content=state["condensed_query"])]

        agent_output = react_agent_executor.invoke({"messages": messages_for_agent})

        all_agent_messages = agent_output.get("messages", [])
        
        # Save verification results
        verification_results = [
            str(msg.content) for msg in all_agent_messages 
            if isinstance(msg, ToolMessage)
        ]
        
        if verification_results:
            existing_results = state.get("verified_results", "")
            new_results_str = "\n".join(verification_results)
            state["verified_results"] = (
                f"{existing_results}\n---\n{new_results_str}" 
                if existing_results else new_results_str
            )
            print(f"--- 📝 VERIFICATION RESULT SAVED TO STATE ---")

        # Get the final AI message from the agent
        final_ai_message = None
        for msg in reversed(all_agent_messages):
            if isinstance(msg, AIMessage):
                final_ai_message = msg
                break
        
        if final_ai_message:
            state["messages"].append(final_ai_message)
        else:
            # Fallback if no AI message found
            fallback_message = AIMessage(content="I've completed the verification process.")
            state["messages"].append(fallback_message)
            
    except Exception as e:
        print(f"---ERROR IN VERIFICATION: {e}---")
        error_message = AIMessage(content=f"I encountered an error during verification: {str(e)}")
        state["messages"].append(error_message)
    
    return state


# --- Memory Configuration ---
memory = MemorySaver()

# --- Graph Definition (Fixed Version) ---
graph = StateGraph(AgentState)

# Add all nodes
graph.add_node("router", router_entry_node)
graph.add_node("condense_query", condense_query)
graph.add_node("retrieve_context", retrieve_context)
graph.add_node("synthesize_answer", synthesize_answer)
graph.add_node("verify_claim", verify_and_respond)
graph.add_node("handle_conversation", handle_conversation)

# Set entry point
graph.set_entry_point("router")

# Add edges
graph.add_edge("router", "condense_query")

# Conditional routing from condense_query
graph.add_conditional_edges(
    "condense_query",
    decide_route,
    {
        "answer_from_context": "retrieve_context",
        "verify_claim": "verify_claim",
        "general_conversation": "handle_conversation",
    },
)

# Conditional routing from retrieve_context
graph.add_conditional_edges(
    "retrieve_context",
    check_relevance,
    {
        "synthesize_answer": "synthesize_answer",
        "verify_claim": "verify_claim",
    },
)

# Terminal edges
graph.add_edge("synthesize_answer", END)
graph.add_edge("verify_claim", END)
graph.add_edge("handle_conversation", END)

# Compile the graph with checkpointer
agent = graph.compile(checkpointer=memory)

# Print the graph structure for debugging
agent.get_graph().print_ascii()