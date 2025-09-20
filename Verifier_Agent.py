# verifier_agent.py (Your original code is fine, just ensure the @tool is active if needed)
import asyncio
from typing import TypedDict, List
from Utils.vector_db import insert_into_vertex_vector_db
from Utils.evidence_chunker import chunk_text_by_paragraph
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage
from prompts import verifier_prompt, content_summarizer_prompt
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory, ChatVertexAI
from langchain_core.tools import tool
from prompts import title_url_inference_prompt

<<<<<<< HEAD
# This function remains unchanged
=======
from vertexai import init
init(project="gen-ai-hackathon-470012", location="us-central1")

# ----------------------
# Helpers
# ----------------------
# (Your helper functions remain the same)
>>>>>>> ac7e9d7 (Sync local files)
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

# ----------------------
# Model settings
# ----------------------

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

summarizer_llm = ChatVertexAI(
    model_name="gemini-2.5-flash-lite", # Note: gemini-2.5-flash-lite may not be a valid model name
    **model_kwargs
)

class AgentState(TypedDict):
    text_news: List[str]
    text_with_evidence: str
    image_path: List[str]
    image_analysis: str
    save_to_vector_db: bool
    verified_results: str

# ----------------------
# Async nodes
# ----------------------
# (Your async nodes remain the same)
async def text_evidence_collection(state: AgentState) -> AgentState:
    print("-> Starting text evidence collection...")
    claims = state["text_news"]
    formatted = ""
    db_tasks = []
    from Tools.Web_Search import web_search_tool as async_web_search_tool
    print('doing web search')
    for claim in claims:
        formatted += f"Claim: {claim}\n"
        print("IT IS DOING STUFF WAIT PLS") 
        search_result = await async_web_search_tool(claim)
        print(search_result)
        print("YOU SNOOZE U LOOSE")
        for result in search_result:
            if result['scrapable']!=False:
                summarization_prompt = content_summarizer_prompt(result["content"])
                response_message = await summarizer_llm.ainvoke(summarization_prompt)
            else:
                title = result.get('title', '')
                url = result.get('url', '')
                prompt = title_url_inference_prompt(title, url)
                response_message = await summarizer_llm.ainvoke(prompt)

            print("DOIING SUMMARY")
            print("DONE SUMMARY")

            
            summary = response_message.content

            print("summary",summary)
            
            formatted += f"Result: {summary}\nTitle: {result['title']}\nURL: {result['url']}\n"

            
            if state["save_to_vector_db"] and result.get('content'):
                metadata = {"claim": claim, "source_url": result["url"], "title": result["title"]}

                chunks = chunk_text_by_paragraph(result['content'])

                for chunk in chunks:
                    task = asyncio.to_thread(insert_into_vertex_vector_db, [chunk], [metadata])
                    db_tasks.append(task)

        
    if db_tasks:
        print(f"-> Starting {len(db_tasks)} DB insertions in the background...")
        await asyncio.gather(*db_tasks)
        print("-> DB insertions complete.")    
    print(formatted)
    state["text_with_evidence"] = formatted
    return state

async def image_analysis(state: AgentState) -> AgentState:
    print("-> Starting image analysis...")
    if not state["image_path"]:
        return state
    db_tasks = []
    from Tools.Reverse_Image_Search import reverse_image_search as async_reverse_image_search
    for idx, image in enumerate(state["image_path"], start=1):
        print(f"-> Starting image analysis for image {idx}: {image}")
        formatted = ""
        search_results = await async_reverse_image_search.ainvoke({"image_path": image})
        if search_results.get("status") == "success":
            print(f"-> Found {len(search_results['scraped_results'])} results from reverse image search.")
            for result in search_results["scraped_results"]:
                if not result["content"]:
                    continue
                response_message = await summarizer_llm.ainvoke(content_summarizer_prompt(result["content"]))
                summary = response_message.content
                formatted += f"Image {idx}\nURL: {result['url']}\nContent: {summary}\n"
                if state["save_to_vector_db"]:
                    chunks = chunk_text_by_paragraph(result["content"])
                    for chunk in chunks:
                        metadata = {"image_id": idx, "source_url": result["url"]}
                        task = asyncio.to_thread(insert_into_vertex_vector_db, [chunk], [metadata])
                        db_tasks.append(task)
        else:
            print("-> No results found from reverse image search.")
        state["image_analysis"] += formatted
        
    if db_tasks:
        print(f"-> Starting {len(db_tasks)} image-related DB insertions in the background...")
        await asyncio.gather(*db_tasks)
        print("-> Image DB insertions complete.")

    return state

async def verify_claims(state: AgentState) -> AgentState:
    print("-> Starting claim verification...")
    verifier = verifier_prompt(state)
    response_message = await summarizer_llm.ainvoke(verifier)
    state["verified_results"] = response_message.content
    return state

# ----------------------
# Graph
# ----------------------
memory = InMemorySaver()
graph = StateGraph(AgentState)
graph.add_node("text_evidence_collection", text_evidence_collection)
graph.add_node("image_analysis", image_analysis)
graph.add_node("verify_claims", verify_claims)
graph.add_edge("text_evidence_collection", "image_analysis")
graph.add_edge("image_analysis", "verify_claims")
graph.set_entry_point("text_evidence_collection")
graph.set_finish_point("verify_claims")
verifier_agent = graph.compile(checkpointer=memory)

# ----------------------
# Tool wrapper
# ----------------------

@tool
async def verifier_tool(
    text_news: List[str],
    image_path: List[str] = None,
    save_to_vector_db: bool = True
) -> str:
    """Balle"""
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
<<<<<<< HEAD
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
=======
    final_state = await verifier_agent.ainvoke(initial_state)
    return final_state["verified_results"]
>>>>>>> ac7e9d7 (Sync local files)
