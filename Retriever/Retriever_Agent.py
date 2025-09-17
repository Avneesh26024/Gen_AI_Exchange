from typing import Dict, List, Optional
from vertexai.preview.generative_models import GenerativeModel, Tool, FunctionDeclaration
from .Retriever import retrieve_chunks
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
# @tool
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

    # Try to extract a function call from the model response. If the model
    # didn't return a function call, fall back to directly using the query.
    function_call = None
    try:
        # response shape may vary between SDK versions; guard against missing fields
        candidate = None
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
        elif hasattr(response, "candidate"):
            candidate = response.candidate

        content_part = None
        if candidate is not None and hasattr(candidate, "content"):
            content = candidate.content
            # content may have .parts or be a direct object
            if hasattr(content, "parts") and content.parts:
                content_part = content.parts[0]
            else:
                content_part = content

        if content_part is not None and hasattr(content_part, "function_call"):
            function_call = content_part.function_call
    except Exception as e:
        print(f"Warning: could not parse model response for function call: {e}")

    api_response = None
    if function_call is not None and getattr(function_call, "args", None):
        args = function_call.args
        # args may be a JSON string or a dict-like object
        if isinstance(args, str):
            import json
            try:
                args = json.loads(args)
            except Exception:
                # treat whole string as query_text
                args = {"query_text": args}

        if isinstance(args, dict):
            # Map common parameter names if needed
            # e.g., model might send {'query': '...', 'filters': {...}}
            if "query_text" not in args and "query" in args:
                args["query_text"] = args.pop("query")
            api_response = evidence_retriever_func(**args)
        else:
            # Fallback to using the raw query
            api_response = evidence_retriever_func(query_text=query)
    else:
        # No function call: just retrieve based on the plain query
        api_response = evidence_retriever_func(query_text=query)

    # Send the result of the function call back to the model so it can formulate a final answer
    # Although in this case, the function result is what we want.
    # For a more complex agent, you might continue the conversation.
    print("--- Retriever Agent finished. ---")
    return api_response

if __name__ == "__main__":
    # Example usage
    query = "drinking water can cure cancer"
    result = retriever_agent(query)
    print(result)