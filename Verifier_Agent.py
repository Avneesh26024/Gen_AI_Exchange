import asyncio
from typing import TypedDict, List, Any, Dict
from Tools.Web_Search import web_search_tool
from Utils.vector_db import insert_into_vertex_vector_db
from Utils.evidence_chunker import chunk_text_by_paragraph
from Tools.Reverse_Image_Search import reverse_image_search
from langgraph.graph import StateGraph
from vertexai.preview.generative_models import GenerativeModel
from prompts import verifier_prompt, content_summarizer_prompt
from langchain_core.tools import tool


model = GenerativeModel("gemini-2.5-flash-lite")
summarizer = GenerativeModel("gemini-2.5-flash-lite")

class AgentState(TypedDict):
    id: int
    text_news: List[str]
    text_with_evidence: str
    image_path: str
    image_analysis: str
    save_to_vector_db: bool
    verified_results: str


async def text_evidence_collection(state: AgentState)-> AgentState:
    """
    Collect text claim evidence from web search and upsert the evidence into Vertex AI Vector DB.
    :param state:
    :return:
    """
    print("-> Starting text evidence collection...")
    claims = state["text_news"]
    formatted = ""
    for claim in claims:
        formatted += f"Claim {claim}:\n"
        search_result = await web_search_tool(claim)
        print(f"-> Found {len(search_result)} results for claim: '{claim}'")
        for result in search_result:
            # Always add the found evidence to the formatted string
            prompt = content_summarizer_prompt(result["content"])
            summary = summarizer.generate_content(prompt)
            formatted += f"Result: {summary}\n"
            formatted += f"Title: {result['title']}\n"
            formatted += f"URL: {result['url']}\n"

            # Only save to vector DB if the flag is True
            if state["save_to_vector_db"]:
                metadata = {
                    "user_id": state["id"],
                    "claim": claim,
                    "source_url": result["url"],
                    "title": result["title"]
                }
                chunks = chunk_text_by_paragraph(result["content"], max_chunk_size=1000, overlap=200)
                for chunk in chunks:
                    insert_into_vertex_vector_db(chunk, metadata)

    state["text_with_evidence"] = formatted
    return state

async def image_analysis(state: AgentState)-> AgentState:
    """
    Analyze the image using reverse image search and collect evidence from the results.
    :param state:
    :return:
    """
    if not state["image_path"]:
        return state
    formatted =""
    search_results = await reverse_image_search.ainvoke({"image_path": state["image_path"]})
    if search_results.get("status") == "success":
        print(f"-> Found {len(search_results['scraped_results'])} results from reverse image search.")
        for result in search_results["scraped_results"]:
            prompt = content_summarizer_prompt(result["content"])
            summary = summarizer.generate_content(prompt)
            formatted += f"URL: {result['url']}\n"
            formatted += f"Content: {summary}\n"
            if state["save_to_vector_db"] is False:
                return state
            chunks = chunk_text_by_paragraph(result["content"])
            for chunk in chunks:
                metadata = {
                    "user_id": state["id"],
                    "source_url": result["url"]
                }
                insert_into_vertex_vector_db(chunk, metadata)
    else:
        print("-> No results found from reverse image search.")

    return state

def verify_claims(state: AgentState) -> AgentState:
    """
    Verify and aggregate claims from the text and image analysis using retrieved evidence.
    Checks if the text_claims and analyzed_image are correlated.
    Return a summary of the verification with citations.
    :param state:
    :return:
    """
    print("-> Starting claim verification...")
    verifier = verifier_prompt(state)
    response = model.generate_content(
        verifier,
        generation_config={"temperature": 0.0}
    )
    state["verified_results"] = response.text
    return state

graph = StateGraph(AgentState)
graph.add_node("text_evidence_collection", text_evidence_collection)
graph.add_node("image_analysis", image_analysis)
graph.add_node("verify_claims", verify_claims)
graph.add_edge("text_evidence_collection", "image_analysis")
graph.add_edge("image_analysis", "verify_claims")
graph.set_entry_point("text_evidence_collection")
graph.set_finish_point("verify_claims")
verifier_agent = graph.compile()
verifier_agent.get_graph().print_ascii()


@tool
async def verifier_tool(text_news: List[str], image_path: str = "", save_to_vector_db: bool = False, user_id: int = 1) -> str:
    """
    A comprehensive tool to verify claims from text and optionally an image.

    This tool orchestrates a multi-step process:
    1.  It collects evidence for each text claim using web searches.
    2.  If an image path is provided, it performs a reverse image search to find related web content.
    3.  Optionally, it can save all collected evidence (text and image-related) into a vector database for future use.
    4.  Finally, it analyzes all the evidence to verify the claims and returns a structured JSON-like string with the results.

    Args:
        text_news (List[str]): A list of strings, where each string is a claim to be verified.
        image_path (str, optional): The local file path to an image related to the claims. Defaults to "".
        save_to_vector_db (bool, optional): If True, all collected evidence will be chunked and stored in the vector database. Defaults to False.
        user_id (int, optional): A unique identifier for the user, used for metadata when saving to the vector database. Defaults to 1.

    Returns:
        str: A JSON-formatted string containing the verification results for each claim, including classification, evidence summary, and final decision.
    """
    initial_state: AgentState = {
        "id": user_id,
        "text_news": text_news,
        "text_with_evidence": "",
        "image_path": image_path,
        "image_analysis": "",
        "save_to_vector_db": save_to_vector_db,
        "verified_results": ""
    }
    final_state = await verifier_agent.ainvoke(initial_state)
    return final_state["verified_results"]


async def main():
    claims = [
        "The Eiffel Tower is located in Berlin.",
        "The COVID-19 vaccine contains microchips for tracking.",
        "Drinking water can cure cancer."
    ]

    # To use the tool, you would typically invoke it like this:
    results = await verifier_tool.ainvoke({
        "text_news": claims,
        "save_to_vector_db": False
    })

    print("-> Final Verification Results:")
    print(results)


if __name__ == "__main__":
    asyncio.run(main())
