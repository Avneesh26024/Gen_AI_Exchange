from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory, ChatVertexAI
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List
from langchain.prompts import ChatPromptTemplate


### CLAIM BRUTEFORCE RETRIEVER

# Assume these are your pre-built agents/tools
import uuid

from Retriever.Retriever_Agent import retriever_agent 

# --- Model and Memory Setup (Unchanged) ---
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
model = ChatVertexAI(model_name="gemini-2.5-flash-lite", **model_kwargs)
memory = MemorySaver()

# --- Agent State (Unchanged) ---
class AgentState(TypedDict):
    messages: List[BaseMessage]
    verified_results: str
    relevant_context: str  
    condensed_query: str
    image_path: List[str]

# --- NODE DEFINITIONS ---

def condense_query(state: AgentState) -> AgentState:
    """Condenses the chat history into a standalone question. Always the first step."""
    print("---NODE: CONDENSING QUERY---")
    user_message = state["messages"][-1].content
    history = state["messages"][:-1]

    if not history:
        condensed_query = user_message
    else:
        # Code to condense query remains the same...
        formatted_history = "\n".join([f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}" for msg in history])
        condensing_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rephrase the follow-up question to be a standalone question."),
            ("human", "Chat History:\n{chat_history}\n\nFollow Up Input: {question}"),
        ])
        response = (condensing_prompt | model).invoke({"chat_history": formatted_history, "question": user_message})
        condensed_query = response.content

    state["condensed_query"] = condensed_query
    print(f"---CONDENSED QUERY: {condensed_query}---")
    return state


def retrieve_context(state: AgentState) -> AgentState:
    """
    Fetches documents from the retriever and formats them into a structured string,
    including metadata like source URL and title for better context.
    """
    print("---NODE: RETRIEVING CONTEXT---")
    query = state["condensed_query"]
    context_result = retriever_agent(query) 
    chunks = context_result.get("retrieved_chunks", [])

    # Handle the case where the retriever finds nothing.
    if not chunks:
        state["relevant_context"] = "No relevant documents were found."
        print("---CONTEXT RETRIEVED (No documents found)---")
        return state

    formatted_context_list = []
    # Loop through each retrieved chunk to format it.
    for i, chunk in enumerate(chunks):
        # Safely extract the text and metadata from the chunk dictionary.
        text = chunk.get("text", "No content available.")
        metadata = chunk.get("filterable_restricts", {})
        
        # Metadata values are lists, so we get the first item or a default.
        source_url = metadata.get("source_url", ["Source not available"])[0]
        title = metadata.get("title", ["Title not available"])[0]

        # Create a clean, formatted block for each document.
        # This is much easier for the LLM to understand.
        context_block = f"""--- Document [{i+1}] ---
Source URL: {source_url}
Source Title: {title}
Content:
{text}"""
        formatted_context_list.append(context_block)

    # Join all the formatted blocks into a single string for the state.
    state["relevant_context"] = "\n\n".join(formatted_context_list)
    print(f"---CONTEXT RETRIEVED ({len(chunks)} documents formatted)---")
    return state

def synthesize_answer(state: AgentState) -> AgentState:
    """Generates the final answer using whatever context is available."""
    print("---NODE: SYNTHESIZING ANSWER---")
    query = state["condensed_query"]
    context = state["relevant_context"]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the user's question based *only* on the provided context."),
        ("human", "Context:\n{context}\n\nUser Question:\n{query}"), 
    ])
    answer_chain = prompt | model
    response_msg = answer_chain.invoke({"context": context, "query": query})
    state["messages"].append(response_msg)
    return state

def verifier(state: AgentState) -> AgentState:
    """The final fallback node when no sufficient context can be found."""
    print("---NODE: VERIFIER---")
    # Wrap the content in an AIMessage object
    response_message = AIMessage(content="I could not find a sufficient answer. The verifier would now take over.")
    state['messages'].append(response_message)
    return state

# --- ROUTER DEFINITIONS ---

# ROUTER 1: Checks if chat history is enough
def check_memory_sufficiency(state: AgentState) -> str:
    """Checks if the conversation history is sufficient to answer the query."""
    print("---ROUTER 1: CHECKING MEMORY SUFFICIENCY---")
    query = state["condensed_query"]
    history_messages = state["messages"][:-1]

    if not history_messages:
        print("---ROUTE: No history. Must retrieve.---")
        return "insufficient"

    formatted_history = "\n".join([f"{msg.type}: {msg.content}" for msg in history_messages])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a query analyzer. Your task is to determine if the Chat History contains enough information to fully answer the User's Query. Respond with only 'yes' or 'no'."),
        ("human", "Chat History:\n{history}\n\nUser's Query:\n{query}"),
    ])
    chain = prompt | model
    response = chain.invoke({"history": formatted_history, "query": query})

    if "yes" in response.content.lower():
        print("---ROUTE: Memory is SUFFICIENT. Synthesizing answer.---")
        # IMPORTANT: We load the history into the context field for the synthesizer
        state["relevant_context"] = formatted_history
        return "sufficient"
    else:
        print("---ROUTE: Memory is INSUFFICIENT. Proceeding to retriever.---")
        return "insufficient"

# ROUTER 2: Checks if retrieved documents are enough
def check_retrieved_context_sufficiency(state: AgentState) -> str:
    """Checks if the newly retrieved context is sufficient to answer the query."""
    print("---ROUTER 2: CHECKING RETRIEVED CONTEXT SUFFICIENCY---")
    query = state["condensed_query"]
    context = state["relevant_context"]

    if not context.strip():
        print("---ROUTE: No context was retrieved. Must verify.---")
        return "insufficient"

    # This is our strict relevance prompt from before
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a highly strict relevance checker. Does the provided CONTEXT directly and fully answer the QUERY? If it's only partially related, say no. Respond with only 'yes' or 'no'."),
        ("human", "CONTEXT:\n{context}\n\nQUERY:\n{query}"),
    ])
    chain = prompt | model
    response = chain.invoke({"context": context, "query": query})

    if "yes" in response.content.lower():
        print("---ROUTE: Retrieved context is SUFFICIENT. Synthesizing answer.---")
        return "sufficient"
    else:
        print("---ROUTE: Retrieved context is INSUFFICIENT. Proceeding to verifier.---")
        return "insufficient"

# --- GRAPH DEFINITION ---

graph = StateGraph(AgentState)

# Add all the nodes
graph.add_node("condense_query", condense_query)
graph.add_node("retrieve_context", retrieve_context)
graph.add_node("synthesize_answer", synthesize_answer)
graph.add_node("verifier", verifier)

# Set the entry point
graph.set_entry_point("condense_query")

# Define the first decision point (Memory Check)
graph.add_conditional_edges(
    "condense_query",
    check_memory_sufficiency,
    {
        "sufficient": "synthesize_answer",
        "insufficient": "retrieve_context",
    },
)

# Define the second decision point (Retrieved Context Check)
graph.add_conditional_edges(
    "retrieve_context",
    check_retrieved_context_sufficiency,
    {
        "sufficient": "synthesize_answer",
        "insufficient": "verifier",
    },
)

# Define the final end points
graph.add_edge("synthesize_answer", END)
graph.add_edge("verifier", END)

# Compile the agent
agent = graph.compile(checkpointer=memory)

# Visualize the graph to confirm the logic
agent.get_graph().print_ascii()


# if __name__ == "__main__":
    
#     # 1. Create a unique ID for this entire conversation session
#     conversation_id = str(uuid.uuid4())
#     print(f"✅ Agent started. Conversation ID: {conversation_id}")
#     print("Type 'exit' or 'quit' to stop.")

#     try:
#         while True:
#             user_input = input("You: ").strip()
#             if not user_input or user_input.lower() in ("exit", "quit", "q"):
#                 break

#             # The input for the agent is just the new message
#             # The checkpointer will handle loading the history
#             agent_input = {"messages": [HumanMessage(content=user_input)]}
            
#             # 2. Pass the config with the unique thread_id here 🔑
#             config = {"configurable": {"thread_id": conversation_id}}
            
#             # Invoke the agent with the input and config
#             final_state = agent.invoke(agent_input, config=config)

#             # The AI's response is the last message in the final state
#             bot_response = final_state["messages"][-1]

#             print(f"Bot: {bot_response.content}")

#     except KeyboardInterrupt:
#         print("\nInterrupted. Exiting.")
#     except Exception as e:
#         print(f"\nAn error occurred: {e}")

