from typing import TypedDict, List, Any, Dict
from Tools.Sync_Wrappers import web_search_tool, reverse_image_search
from Utils.vector_db import insert_into_vertex_vector_db
from Utils.evidence_chunker import chunk_text_by_paragraph
from langgraph.graph import StateGraph
from langchain_core.messages import AIMessage
from google.genai import types
from prompts import verifier_prompt, content_summarizer_prompt, stance_prompt
from google import genai
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory, ChatVertexAI
from langchain_core.tools import tool
import json
from urllib.parse import urlparse
from langchain_google_vertexai import VertexAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from Tools.Visual_Tool import analyze_image

embedding_model = VertexAIEmbeddings(model_name="text-embedding-004")
client = genai.Client(
  vertexai=True, location="us-central1"
)

# Define domain-based credibility and tags
DOMAIN_CREDIBILITY = {
    # Trusted international news
    "reuters.com": (0.95, "news_trusted"),
    "bbc.com": (0.9, "news_trusted"),
    "cnn.com": (0.9, "news_trusted"),
    "theguardian.com": (0.9, "news_trusted"),
    "nytimes.com": (0.9, "news_trusted"),
    "wikipedia.org": (0.9, "reference"),
    "who.int": (1.0, "health_official"),

    # Indian government / official
    "gov.in": (1.0, "gov_official"),
    "india.gov.in": (1.0, "gov_official"),
    "mohfw.gov.in": (1.0, "health_official"),

    # Blogs / news aggregators
    "medium.com": (0.6, "blog"),
    "techcrunch.com": (0.7, "blog_news"),
    "mashable.com": (0.6, "blog_news"),

    # Forums / community content
    "reddit.com": (0.3, "forum"),
    "quora.com": (0.3, "forum"),
    "stackexchange.com": (0.5, "forum"),

    # Social media
    "twitter.com": (0.2, "social_media"),
    "facebook.com": (0.2, "social_media"),
    "instagram.com": (0.2, "social_media"),

    # Clickbait / suspicious
    "randomtravelblog.com": (0.2, "clickbait"),
    "unknownnews.com": (0.1, "unknown"),
}

DEFAULT_CREDIBILITY = (0.3, "unknown")

def assign_credibility(url: str) -> dict:
    """
    Assigns a credibility score and tag to a given URL using simple substring matching.

    Args:
        url (str): The URL of the source.

    Returns:
        dict: {"credibility": float, "tag": str}
    """
    domain = urlparse(url).netloc.lower()
    for key, (score, tag) in DOMAIN_CREDIBILITY.items():
        if key in domain:
            return {"credibility": score, "tag": tag}
    return {"credibility": DEFAULT_CREDIBILITY[0], "tag": DEFAULT_CREDIBILITY[1]}



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
    "max_output_tokens": 2048,
    "top_p": 0.95,
    "top_k": None,
    "safety_settings": safety_settings,
}
verifier_kwargs = {
    "temperature": 0.28,
    "max_output_tokens": 10000,
    "top_p": 0.95,
    "top_k": None,
    "safety_settings": safety_settings,
}

# The summarizer_llm and AgentState remain unchanged
summarizer_llm = ChatVertexAI(
    model_name="gemini-2.5-flash-lite", # Using a stable model name
    **model_kwargs
)
llm = ChatVertexAI(
    model_name="gemini-2.5-flash-lite", # Using a stable model name
    **verifier_kwargs
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
    results_by_claim = []

    for claim in claims:
        print(f"\n-> Processing claim: {claim}")
        search_results = web_search_tool(claim)
        print(f"-> Found {len(search_results)} results for claim: '{claim}'")

        claim_evidence_list = []  # store evidence for this claim

        for result in search_results:
            if not result.get("scrapable", False):
                continue

            content = result.get("content", "")
            if not content:
                continue

            # 1️⃣ Summarize content
            summarization_prompt = content_summarizer_prompt(content)
            response_message = summarizer_llm.invoke(summarization_prompt)
            summary = response_message.content.strip()

            # 2️⃣ Stance detection
            stance_response = llm.invoke(stance_prompt(claim, content))
            stance_text = stance_response.content.strip().lower()
            if stance_text == "supports":
                stance = 1
            elif stance_text == "refutes":
                stance = -1
            else:
                stance = 0

            # 3️⃣ Credibility
            credibility = assign_credibility(result.get("url", ""))

            # 4️⃣ Relevance (using embeddings)
            embd_claim = embedding_model.embed_query(claim)
            embd_evidence = embedding_model.embed_query(summary)
            similarity = cosine_similarity([embd_claim], [embd_evidence])[0][0]
            relevance = (similarity + 1) / 2  # normalize

            # 5️⃣ Recency (if you add publish_date later)
            recency = result.get("publish_date", None)

            # 6️⃣ Store evidence dict
            evidence_item = {
                "summary": summary,
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "stance": stance_text.upper(),
                "stance_score": stance,
                "credibility": credibility["credibility"],
                "relevance": relevance,
                "recency": recency
            }
            claim_evidence_list.append(evidence_item)

            # Insert into vector DB if required
            if state["save_to_vector_db"]:
                metadata = {
                    "claim": claim,
                    **evidence_item  # include all fields
                }
                if state["save_to_vector_db"]:
                    chunks = chunk_text_by_paragraph(content)
                    insert_into_vertex_vector_db(chunks, metadata)

        # 🔑 Confidence score for this claim
        if claim_evidence_list:
            total_weight = 0
            for ev in claim_evidence_list:
                # simple confidence formula: stance * credibility * relevance
                weight = ev["stance_score"] * ev["credibility"] * ev["relevance"]
                total_weight += weight
            confidence_score = total_weight / len(claim_evidence_list)
        else:
            confidence_score = 0

        results_by_claim.append({
            "claim": claim,
            "evidence": claim_evidence_list,
            "confidence_score": confidence_score
        })

    # Store structured results in state
    state["text_with_evidence"] = results_by_claim
    return state


# This synchronous node can remain as is. LangGraph handles mixed async/sync nodes.
def image_analysis(state: AgentState) -> AgentState:
    print("-> Starting image analysis...")
    if not state["image_path"]:
        return state

    number = 0
    for image_path in state["image_path"]:
        number += 1
        print(f"-> Analyzing image {number}: {image_path}")
        formatted = ""

        # 1️⃣ Analyze image for OCR, labels, landmarks, logos, objects
        analysis_data = analyze_image(image_path)
        ocr_text = " ".join([t["description"] for t in analysis_data.get("texts", [])])
        labels_text = ", ".join([l["description"] for l in analysis_data.get("labels", [])])
        landmarks_text = ", ".join([l["description"] for l in analysis_data.get("landmarks", [])])
        logos_text = ", ".join([l["description"] for l in analysis_data.get("logos", [])])
        objects_text = ", ".join([f"{o['name']}({o['confidence']:.2f})" for o in analysis_data.get("objects", [])])

        # 2️⃣ Generate image caption using Gemini
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        caption_response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
                "Generate a concise descriptive caption for this image."
            ]
        )
        image_caption = caption_response.text.strip()

        # Combine OCR, labels, caption, landmarks, logos, objects
        text_for_analysis = " ".join(filter(None, [ocr_text, labels_text, image_caption, landmarks_text, logos_text, objects_text]))
        if not text_for_analysis:
            text_for_analysis = "No text, labels, caption, landmarks, logos, or objects detected"

        # 3️⃣ Reverse image search (synchronous)
        search_results = reverse_image_search(image_path)
        if search_results.get("status") != "success" or not search_results.get("scraped_results"):
            print("-> No results found from reverse image search.")
            continue

        evidence_list = []
        for result in search_results["scraped_results"]:
            if not result.get("content"):
                print(f"-> Skipping URL with no content: {result.get('url')}")
                continue

            # 4️⃣ Summarize content
            summary = summarizer_llm.invoke(content_summarizer_prompt(result["content"])).content

            # 5️⃣ Compute stance
            stance_text = llm.invoke(stance_prompt(text_for_analysis, result["content"])).content.strip().lower()
            stance = 1 if stance_text == "supports" else -1 if stance_text == "refutes" else 0

            # 6️⃣ Compute credibility
            credibility = assign_credibility(result["url"])["credibility"]

            # 7️⃣ Compute relevance using embeddings
            embd_image_text = embedding_model.embed_query(text_for_analysis)
            embd_evidence = embedding_model.embed_query(summary)
            similarity = cosine_similarity([embd_image_text], [embd_evidence])[0][0]
            relevance = (similarity + 1) / 2  # Normalize 0-1

            # 8️⃣ Store evidence
            evidence_list.append({
                "url": result.get("url"),
                "title": result.get("title", ""),
                "summary": summary,
                "stance": stance,
                "credibility": credibility,
                "relevance": relevance
            })

            # 🔟 Insert chunks into vector DB
            if state["save_to_vector_db"]:
                metadata = {
                    "image_id": number,
                    "ocr_text": ocr_text,
                    "labels_text": labels_text,
                    "caption": image_caption,
                    "landmarks_text": landmarks_text,
                    "logos_text": logos_text,
                    "objects_text": objects_text,
                    "source_url": result.get("url"),
                    "title": result.get("title", ""),
                    "summary": summary,
                    "stance": stance,
                    "credibility": credibility,
                    "relevance": relevance
                }
                if state["save_to_vector_db"]:
                    chunks = chunk_text_by_paragraph(result["content"])
                    insert_into_vertex_vector_db(chunks, metadata)

        # 1️⃣1️⃣ Compute overall confidence score (without recency)
        confidence = 0
        total_weight = 0
        for ev in evidence_list:
            weight = ev["credibility"] * ev["relevance"]  # recency removed
            confidence += ev["stance"] * weight
            total_weight += weight
        confidence_score = confidence / total_weight if total_weight > 0 else 0

        # 1️⃣2️⃣ Add formatted info
        formatted += f"Image {number}: {image_path}\n"
        formatted += f"OCR + Labels + Caption + Landmarks + Logos + Objects: {text_for_analysis}\n"
        formatted += f"Confidence Score: {confidence_score:.3f}\n"
        for ev in evidence_list:
            formatted += f"- URL: {ev['url']}, Stance: {ev['stance']}, Credibility: {ev['credibility']:.2f}, Relevance: {ev['relevance']:.2f}\n"

        state["image_analysis"] += formatted

    return state


def verify_claims(state: AgentState) -> AgentState:
    print("-> Starting claim verification and synthesis...")

    # Generate the detailed prompt using the state
    verifier = verifier_prompt(state)

    # Call the LLM to get the JSON output
    response_message = llm.invoke(verifier)  # Using llm which is configured for this type of task
    raw_output = response_message.content.strip()

    # Clean the output in case the LLM wraps it in markdown
    if raw_output.startswith("```json"):
        raw_output = raw_output[7:]
    if raw_output.endswith("```"):
        raw_output = raw_output[:-3]

    # Try to parse the output as JSON
    try:
        parsed_json = json.loads(raw_output)
        # If successful, store the clean, parsed JSON object
        state["verified_results"] = parsed_json
    except json.JSONDecodeError:
        print("-> ERROR: LLM output was not valid JSON. Storing raw output.")
        # If parsing fails, store the raw string for debugging
        state["verified_results"] = {"error": "Failed to parse LLM output as JSON", "raw_output": raw_output}

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
def verifier_tool(text_news: List[str], image_path: List[str] = None, save_to_vector_db: bool = False) -> str:
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
    final_results = verifier_tool(text_news=claims, image_path=[], save_to_vector_db=False)
    print("\n-> Final Verification Results:")
    print(final_results)


# CHANGED: Use asyncio.run() to execute the async main function
if __name__ == "__main__":
    main()