import uuid
from typing import TypedDict, Sequence, Annotated

from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages

# --- 1. Define the state schema (Unchanged) ---
class ChatState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# --- 2. Memory setup (Unchanged) ---
memory = MemorySaver()

# --- 3. LLM setup (Unchanged) ---
llm = ChatOllama(model='llama3')

# --- 4. Node function (Unchanged) ---
def simple_chat_agent(state: ChatState):
    print("-> Calling Chat Agent")
    # The agent receives the full message history from the checkpointer
    # and passes it to the LLM.
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# --- 5. Build the graph (Simplified) ---
graph_builder = StateGraph(ChatState)
graph_builder.add_node("chat_agent", simple_chat_agent)

# The graph now has a simple, linear flow.
# It starts, calls the agent, and then immediately ends.
graph_builder.add_edge(START, "chat_agent")
graph_builder.add_edge("chat_agent", END)

# --- 6. Compile the app (Unchanged) ---
app = graph_builder.compile(checkpointer=memory)

# --- 7. Run the interactive chat loop (Unchanged) ---
if __name__ == "__main__":
    # A unique ID for this specific conversation
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    print("Chatbot is ready! Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ").strip()
        
        # Check for exit condition before invoking the graph
        if user_input.lower() in ["exit", "quit"]:
            print("Bot: Goodbye!")
            break
        
        if not user_input:
            continue

        # Prepare the input for the graph
        inputs = {"messages": [HumanMessage(content=user_input)]}
        
        # Invoke the graph. It will run one full turn (user -> AI).
        result = app.invoke(inputs, config=config)
        
        # Get the AI's response (which is the last message in the list)
        ai_response = result["messages"][-1]
        print(f"Bot: {ai_response.content}")