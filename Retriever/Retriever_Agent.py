from typing import Dict, List, Optional
from vertexai.preview.generative_models import GenerativeModel, Tool, FunctionDeclaration
from Retriever import retrieve_chunks
from prompts import retriever_prompt
from langchain_core.tools import tool

# --- Step 1: Define the function that will be wrapped as a tool for the LLM ---
def evidence_retriever_func(query_text: str, top_k: int = 3, filters: Optional[Dict[str, List[str]]] = None):
    """
    Retrieves evidence chunks from the vector index based on a query and optional filters.
    Args:
        query_text: The text to search for.
        top_k: The number of results to return.
        filters: A dictionary for metadata filtering (e.g., {"region": ["North India"]}).
    """
    print(f"--- Executing evidence_retriever_func with query: '{query_text}' and filters: {filters} ---")
    chunks = retrieve_chunks(query_text=query_text, top_k=top_k, filters=filters)
    return {"retrieved_chunks": chunks}

# --- Step 2: Create the tool and model for the agent ---
retriever_tool_for_llm = Tool(function_declarations=[FunctionDeclaration.from_func(evidence_retriever_func)])

# It's recommended to use a powerful model for this agent to understand the query and use the tool effectively.
model = GenerativeModel(
    "gemini-2.5-flash-lite",
    system_instruction=retriever_prompt(),
    tools=[retriever_tool_for_llm]
)

# --- Step 3: Create the LangGraph-compatible tool ---
@tool
def retriever_agent(query: str) -> Dict:
    """
    A tool that acts as an intelligent retriever agent.
    It takes a user query, uses a powerful LLM to analyze it,
    and then retrieves relevant chunks from a vector database.
    """
    print(f"--- Retriever Agent started with query: '{query}' ---")
    # Start a chat with the model
    chat = model.start_chat()

    # Send the user's query. The model will likely respond with a tool call.
    response = chat.send_message(query)

    # The response will contain a function call to our evidence_retriever_func
    function_call = response.candidates[0].content.parts[0].function_call

    # Now we need to call the function with the arguments provided by the model
    # and get the results.
    api_response = evidence_retriever_func(**function_call.args)

    # Send the result of the function call back to the model so it can formulate a final answer
    # Although in this case, the function result is what we want.
    # For a more complex agent, you might continue the conversation.
    print("--- Retriever Agent finished. ---")
    return api_response

if __name__ == "__main__":
    # Example usage
    query = "What are some popular North Indian snacks?"
    result = retriever_agent.invoke({"query": query})
    print(result)