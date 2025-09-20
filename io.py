import uuid
from typing import TypedDict, List, Annotated, Sequence

from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_vertexai import (ChatVertexAI, HarmBlockThreshold,
                                       HarmCategory)
# I will use your MemorySaver, as requested.
from Retriever.Retriever_Agent import retriever_agent

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages


# --- MOCK RETRIEVER FOR TESTING ---



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
# Using your requested MemorySaver. NOTE: This is a non-functional base class.
# For working memory, this should be:
# from langgraph.checkpoint.sqlite import SqliteSaver
# memory = SqliteSaver.from_conn_string(":memory:")
memory = MemorySaver()


# --- Agent State ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    relevant_context: str
    condensed_query: str
    image_path: List[str]


# --- Node Definitions ---

def condense_query(state: AgentState) -> AgentState:
    """Condenses chat history into a standalone question."""
    # This node is unchanged
    print("---NODE: CONDENSING QUERY---")
    user_message = state["messages"][-1].content
    history = state["messages"][:-1]
    if not history:
        state["condensed_query"] = user_message
        return state
    formatted_history = "\n".join(
        [f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}" for msg in history]
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the conversation and a follow-up, rephrase the follow-up into a standalone question."),
        ("human", "Chat History:\n{chat_history}\n\nFollow Up: {question}"),
    ])
    response = (prompt | model).invoke({"chat_history": formatted_history, "question": user_message})
    state["condensed_query"] = response.content
    return state

def decide_route(state: AgentState) -> str:
    """Routes between using the retriever or handling a general conversation."""
    # This node is unchanged
    print("---ROUTER: DECIDING ROUTE---")
    query = state["condensed_query"]
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", """Classify the user's request:
- 'use_retriever': For factual questions about topics, people, or events.
- 'general_conversation': For greetings, statements, or questions about the conversation."""),
        ("human", "User request: {user_query}"),
    ])
    route = (router_prompt | model).invoke({"user_query": query})
    if "use_retriever" in route.content.strip().lower():
        return "use_retriever"
    else:
        return "general_conversation"

def retrieve_context(state: AgentState) -> AgentState:
    """Retrieves context from the vector database."""
    # This node is unchanged
    print("---NODE: RETRIEVE CONTEXT---")
    query = state["condensed_query"]
    context_result = retriever_agent(query)
    chunks = context_result.get("retrieved_chunks", [])
    state["relevant_context"] = "\n\n".join([chunk.get("text", "") for chunk in chunks]) or "No relevant information found."
    return state

# --- NEW ROUTER to check if the context is good enough ---
def check_context_sufficiency(state: AgentState) -> str:
    """Checks if the retrieved context is sufficient to answer the question."""
    print("---ROUTER: CHECKING CONTEXT SUFFICIENCY---")
    context = state["relevant_context"]
    query = state["condensed_query"]

    if "No relevant information found" in context or not context.strip():
        print("---ROUTE: Context is empty. Calling verifier.---")
        return "insufficient"

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict relevance checker. Based on the CONTEXT provided, can you confidently and completely answer the QUESTION? Respond with only 'yes' or 'no'."),
        ("human", "CONTEXT:\n{context}\n\nQUESTION:\n{question}"),
    ])
    
    chain = prompt | model
    response = chain.invoke({"context": context, "question": query})
    
    if "yes" in response.content.strip().lower():
        print("---ROUTE: Context is SUFFICIENT. Synthesizing answer.---")
        return "sufficient"
    else:
        print("---ROUTE: Context is INSUFFICIENT. Calling verifier.---")
        return "insufficient"

def synthesize_answer(state: AgentState) -> AgentState:
    """Generates an answer based on retrieved context."""
    # This node is unchanged
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
    # This node is unchanged
    print("---NODE: HANDLE CONVERSATION---")
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
        "question": state["condensed_query"]
    })
    state["messages"].append(response_msg)
    return state

# --- NEW NODE that gets called when context is insufficient ---
def verifier(state: AgentState) -> AgentState:
    """This node is called when the retriever fails to find sufficient information."""
    print("---NODE: VERIFIER---")
    message = "Verifier should be called because the retrieved context was not sufficient."
    state["messages"].append(AIMessage(content=message))
    return state

# --- Graph Definition (UPDATED) ---
graph = StateGraph(AgentState)
graph.add_node("condense_query", condense_query)
graph.add_node("retrieve_context", retrieve_context)
graph.add_node("synthesize_answer", synthesize_answer)
graph.add_node("handle_conversation", handle_conversation)
graph.add_node("verifier", verifier) # Add the new node

graph.set_entry_point("condense_query")
graph.add_conditional_edges("condense_query", decide_route, {
    "use_retriever": "retrieve_context",
    "general_conversation": "handle_conversation",
})

# The OLD direct edge is replaced with this new conditional logic
# graph.add_edge("retrieve_context", "synthesize_answer")
graph.add_conditional_edges(
    "retrieve_context",
    check_context_sufficiency,
    {
        "sufficient": "synthesize_answer",
        "insufficient": "verifier"
    }
)

graph.add_edge("synthesize_answer", END)
graph.add_edge("handle_conversation", END)
graph.add_edge("verifier", END) # The verifier path also ends the turn

agent = graph.compile(checkpointer=memory)

# --- Main Chat Loop ---
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