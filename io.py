import uuid
from typing import TypedDict, List, Annotated, Sequence # FIX: Import Annotated and Sequence

from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_vertexai import (ChatVertexAI, HarmBlockThreshold,
                                       HarmCategory)
# FIX: Import a working checkpointer and the 'add_messages' helper
from Retriever.Retriever_Agent import retriever_agent
from langgraph.checkpoint.memory import MemorySaver

from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages


# --- MOCK RETRIEVER FOR TESTING ---
# This is a fake retriever so you can test the graph's logic without errors.
# Replace this with your actual 'from Retriever.Retriever_Agent import retriever_agent' when ready.

# --- Model and Memory Configuration ---
safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
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

# FIX 1 (cont.): Use the working SqliteSaver instead of the non-functional base class.
memory = MemorySaver()


# --- Agent State ---
class AgentState(TypedDict):
    # FIX 2: The 'messages' field MUST be wrapped in Annotated for memory to work.
    messages: Annotated[Sequence[BaseMessage], add_messages]
    relevant_context: str
    condensed_query: str
    image_path: List[str] # Keeping for future use

# --- Node Definitions (Your code is unchanged here) ---

def condense_query(state: AgentState) -> AgentState:
    """Condenses chat history into a standalone question."""
    print("---NODE: CONDENSING QUERY---")
    print("--------------------------------------------------")
    print(f"DEBUG: Agent received {len(state['messages'])} message(s) in its state.")
    for i, msg in enumerate(state['messages']):
        print(f"  -> Message [{i}]: Type={msg.type}, Content='{msg.content}'")
    print("--------------------------------------------------")
    user_message = state["messages"][-1].content
    history = state["messages"][:-1]
    if not history:
        print("VERDICT: No history found. Treating as the first message.")
        state["condensed_query"] = user_message
        return state
    print("VERDICT: History found. Rephrasing query based on context.")
    formatted_history = "\n".join(
        [f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}" for msg in history]
    )
    condensing_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the conversation and a follow-up question, rephrase the follow-up into a standalone question."),
        ("human", "Chat History:\n{chat_history}\n\nFollow Up Input: {question}"),
    ])
    response = (condensing_prompt | model).invoke({"chat_history": formatted_history, "question": user_message})
    state["condensed_query"] = response.content
    print(f"---CONDENSED QUERY: {response.content}---")
    return state

def decide_route(state: AgentState) -> str:
    """Routes between using the retriever or handling a general conversation."""
    print("---ROUTER: DECIDING ROUTE---")
    query = state["condensed_query"]
    
    # This improved prompt gives clearer instructions to the router LLM.
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at routing user requests. Classify the user's request into one of two categories:

1.  **use_retriever**: The user is asking a clear factual question about a topic, person, place, or event that likely requires searching a knowledge base.
    Examples: "Who is the president?", "What is LangGraph?", "what is my first prompt"

2.  **general_conversation**: The user is having a regular conversation. This includes greetings, statements, or giving information.
    Examples: "hi", "thanks that was helpful", "my name is Puneet"
"""),
        ("human", "User request: {user_query}"),
    ])
    
    route = (router_prompt | model).invoke({"user_query": query})
    
    if "use_retriever" in route.content.strip().lower():
        print("---ROUTE: To RETRIEVE_CONTEXT---")
        return "use_retriever"
    else:
        print("---ROUTE: To GENERAL_CONVERSATION---")
        return "general_conversation"

def retrieve_context(state: AgentState) -> AgentState:
    """Retrieves context from the vector database."""
    print("---NODE: RETRIEVE CONTEXT---")
    query = state["condensed_query"]
    context_result = retriever_agent(query)
    chunks = context_result.get("retrieved_chunks", [])
    state["relevant_context"] = "\n\n".join([chunk.get("text", "") for chunk in chunks]) or "No relevant information found."
    print("---CONTEXT RETRIEVED---")
    return state

def synthesize_answer(state: AgentState) -> AgentState:
    """Generates an answer based on retrieved context."""
    print("---NODE: SYNTHESIZE ANSWER---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question based *only* on the provided context."),
        ("human", "Context:\n{context}\n\nQuestion:\n{query}"),
    ])
    response_msg = (prompt | model).invoke({"context": state["relevant_context"], "query": state["condensed_query"]})
    state["messages"].append(response_msg)
    return state

def handle_conversation(state: AgentState) -> AgentState:
    """Handles conversational turns by explicitly passing the full chat history."""
    print("---NODE: HANDLE CONVERSATION---")
    query = state["condensed_query"]
    all_messages = state["messages"]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful and conversational AI assistant. Use the provided chat history to answer the user's question."),
        ("user", "Here is the chat history:\n{history}\n\nBased on that history, please answer this question:\n{question}")
    ])
    history_string = "\n".join(
        f"{msg.type.upper()}: {msg.content}" for msg in all_messages
    )
    chain = prompt | model
    response_msg = chain.invoke({
        "history": history_string,
        "question": query
    })
    state["messages"].append(response_msg)
    return state

# --- Graph Definition (Your code is unchanged here) ---
graph = StateGraph(AgentState)
graph.add_node("condense_query", condense_query)
graph.add_node("retrieve_context", retrieve_context)
graph.add_node("synthesize_answer", synthesize_answer)
graph.add_node("handle_conversation", handle_conversation)

graph.set_entry_point("condense_query")
graph.add_conditional_edges("condense_query", decide_route, {
    "use_retriever": "retrieve_context",
    "general_conversation": "handle_conversation",
})
graph.add_edge("retrieve_context", "synthesize_answer")
graph.add_edge("synthesize_answer", END)
graph.add_edge("handle_conversation", END)

agent = graph.compile(checkpointer=memory)


# --- Main Chat Loop (Your code is unchanged here) ---
if __name__ == "__main__":
    conversation_id = str(uuid.uuid4())
    print(f"✅ Agent started. Conversation ID: {conversation_id}")
    print("Type 'exit' or 'quit' to stop.")
    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input or user_input.lower() in ("exit", "quit", "q"):
                break
            agent_input = {"messages": [HumanMessage(content=user_input)]}
            config = {"configurable": {"thread_id": conversation_id}}
            final_state = agent.invoke(agent_input, config=config)
            last_message = final_state["messages"][-1]
            print(f"Bot: {last_message.content}")
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")