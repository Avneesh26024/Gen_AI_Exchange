from typing import TypedDict, List, Any, Dict
from Tools.Sync_Wrappers import web_search_tool, reverse_image_search
from Utils.vector_db import insert_into_vertex_vector_db
from Utils.evidence_chunker import chunk_text_by_paragraph
from langgraph.graph import StateGraph
from langchain_core.messages import AIMessage
from prompts import verifier_prompt, content_summarizer_prompt
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory, ChatVertexAI
from langchain_core.tools import tool


# This function remains unchanged
def extract_final_ai_message(response: dict) -> str:
    messages = response.get("messages", [])
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content:
            return m.content
    for m in reversed(messages):
        if isinstance(m, dict):
            if m.get("type") == "constructor" and m.get("id", [])[-1] == "AIMessage":
                content = m.get("kwargs", {}).get("content")
                if content:
                    return content
    return None


# These settings remain unchanged
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
summarizer_llm = ChatVertexAI(
    model_name="gemini-2.5-flash-lite", # Using a stable model name
    **model_kwargs
)


class AgentState(TypedDict):
    text_news: List[str]
    text_with_evidence: str
    image_path: List[str]
    image_analysis: str
    save_to_vector_db: bool
    verified_results: str


# This function is now correctly defined as async
def text_evidence_collection(state: AgentState) -> AgentState:
    print("-> Starting text evidence collection...")
    claims = state["text_news"]
    formatted = ""
    for claim in claims:
        formatted += f"Claim: {claim}\n"
        # The async web_search_tool is awaited
        search_result = web_search_tool(claim)
        print(f"-> Found {len(search_result)} results for claim: '{claim}'")
        for result in search_result:
            if not result["scrapable"]:
                continue
            # Note: .invoke() on an LLM is synchronous. For full async, use .ainvoke()
            summarization_prompt = content_summarizer_prompt(result["content"])
            response_message = summarizer_llm.invoke(summarization_prompt)
            summary = response_message.content
            formatted += f"Result: {summary}\n"
            formatted += f"Title: {result['title']}\n"
            formatted += f"URL: {result['url']}\n"
            if state["save_to_vector_db"]:
                metadata = {"claim": claim, "source_url": result["url"], "title": result["title"]}
                chunks = chunk_text_by_paragraph(result["content"])
                for chunk in chunks:
                    # Assuming insert_into_vertex_vector_db can be async or is fast
                    insert_into_vertex_vector_db([chunk], [metadata])
    state["text_with_evidence"] = formatted
    return state


# This synchronous node can remain as is. LangGraph handles mixed async/sync nodes.
def image_analysis(state: AgentState) -> AgentState:
    print("-> Starting image analysis...")
    number = 0
    if not state["image_path"]:
        return state
    for images in state["image_path"]:
        number += 1
        print(f"-> Starting image analysis for image: {images}{number}...")
        formatted = ""
        search_results = reverse_image_search(images)
        if search_results.get("status") == "success":
            print(f"-> Found {len(search_results['scraped_results'])} results from reverse image search.")
            for result in search_results["scraped_results"]:
                if not result["content"]:
                    print(f"-> Skipping summarization for URL with no content: {result['url']}")
                    continue
                response_message = summarizer_llm.invoke(content_summarizer_prompt(result["content"]))
                summary = response_message.content
                formatted += f"Image {number}\nURL: {result['url']}\nContent: {summary}\n"
                if state["save_to_vector_db"]:
                    chunks = chunk_text_by_paragraph(result["content"])
                    for chunk in chunks:
                        metadata = {"image_id": number, "source_url": result["url"]}
                        insert_into_vertex_vector_db([chunk], [metadata])
        else:
            print("-> No results found from reverse image search.")
        state["image_analysis"] += formatted
    return state


def verify_claims(state: AgentState) -> AgentState:
    print("-> Starting claim verification...")
    verifier = verifier_prompt(state)
    response_message = summarizer_llm.invoke(verifier)
    status = response_message.content
    state["verified_results"] = status
    return state


# Graph definition remains the same
graph = StateGraph(AgentState)
graph.add_node("text_evidence_collection", text_evidence_collection)
graph.add_node("image_analysis", image_analysis)
graph.add_node("verify_claims", verify_claims)
graph.add_edge("text_evidence_collection", "image_analysis")
graph.add_edge("image_analysis", "verify_claims")
graph.set_entry_point("text_evidence_collection")
graph.set_finish_point("verify_claims")
verifier_agent = graph.compile()


# --- Tool wrapper ---
# CHANGED: The tool wrapper is now a sync function
@tool
def verifier_tool(text_news: List[str], image_path: List[str] = None, save_to_vector_db: bool = True) -> str:
    """
        Fact-checks and verifies a list of claims using web search evidence.

        Use this tool to determine the factual accuracy of statements, news headlines,
        or user questions like "Is it true that...?". It is the primary tool for
        debunking misinformation or confirming facts. The tool returns a structured
        JSON output that you can parse to answer the user's query.

        Args:
            text_news (List[str]): A list of strings, where each string is a
                single, complete claim to be verified. You MUST provide the input
                as a list of strings.
                Example: ["The Eiffel Tower is located in Berlin.", "The 2028 Olympics will be in Los Angeles."]

        Returns:
            str: A JSON string representing a list of verification objects. Each
                object contains the original 'claim', a 'classification' (e.g.,
                'FAKE', 'TRUE'), a 'text_evidence_summary', and a 'final_decision'.
                You must parse this JSON to extract the specific findings for your final answer.
        """
    if image_path is None:
        image_path = []
    initial_state: AgentState = {
        "text_news": text_news,
        "text_with_evidence": "",
        "image_path": image_path,
        "image_analysis": "",
        "save_to_vector_db": save_to_vector_db,
        "verified_results": ""
    }
    # CHANGED: Use .invoke() for synchronous graph execution
    final_state = verifier_agent.invoke(initial_state)
    return final_state["verified_results"]


# --- Test main ---
def main():
    claims = [
        "The Eiffel Tower is located in Berlin.",
        "The COVID-19 vaccine contains microchips for tracking.",
        "Drinking water can cure cancer."
    ]
    # CHANGED: Await the async tool function
    final_results = verifier_tool(text_news=claims, image_path=[], save_to_vector_db=True)
    print("\n-> Final Verification Results:")
    print(final_results)


# CHANGED: Use asyncio.run() to execute the async main function
if __name__ == "__main__":
    main()