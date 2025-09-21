from typing import TypedDict, List, Dict
from Tools.Sync_Wrappers import web_search_tool, reverse_image_search
from Utils.vector_db import insert_into_vertex_vector_db
from Utils.evidence_chunker import chunk_text_by_paragraph
from langgraph.graph import StateGraph
from langchain_core.messages import AIMessage
from google.genai import types
from prompts import verifier_prompt, content_summarizer_prompt, stance_prompt, validation_prompt
from google import genai
from datetime import datetime, timezone
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory, ChatVertexAI
from langchain_core.tools import tool
import json
from Retriever.Retriever import retrieve_chunks
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

validation_safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}
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
    "temperature": 0.0,
    "max_output_tokens": 10000,
    "top_p": 0.95,
    "top_k": None,
    "safety_settings": safety_settings,
}


# Create a new LLM instance just for the validation task
validation_llm = ChatVertexAI(
    model_name="gemini-2.5-flash-lite", # Using a stable model name
    safety_settings=validation_safety_settings,
    temperature=0.0 # Keep it deterministic
)

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
    already_verified: List[Dict[str, str]]  # List of {"claim": "...", "result": "..."}
    unverified_claims: List[str]
    image_analysis: str
    save_to_vector_db: bool
    verified_results: str


# In your main agent file

# In your main agent file

def router(state: AgentState) -> AgentState:
    """
    Identifies claims that are similar to previously verified ones and passes
    their details (matched claim text and source URLs) for evidence reprocessing.
    """
    print("-> 🧠 Router: Checking for previously verified claims...")
    input_claims = state["text_news"]
    verified_list = []
    unverified_list = []
    filters = {"doc_type": ["verified_claim"]}

    for claim in input_claims:
        retrieved_candidates = retrieve_chunks(claim, top_k=3, filters=filters)
        match_found = False
        if retrieved_candidates:
            prompt = validation_prompt(claim, retrieved_candidates)
            try:
                # --- THIS IS THE CRITICAL LINE ---
                # Ensure it uses 'validation_llm', not 'llm'
                response = validation_llm.invoke(prompt)
                # ---

                validation_result = json.loads(response.content)

                if validation_result.get("match_found"):
                    matched_id = validation_result.get("matched_id")
                    matched_claim_data = retrieved_candidates[matched_id - 1]

                    matched_claim_text = matched_claim_data.get("text")
                    source_urls = matched_claim_data["filterable_restricts"].get("source_urls", [])

                    if matched_claim_text and source_urls:
                        print(f"✅ Cache HIT for claim: '{claim}'. Found similar claim: '{matched_claim_text}'")
                        verified_list.append({
                            "claim": claim,
                            "matched_claim": matched_claim_text,
                            "source_urls": source_urls
                        })
                        match_found = True

            except (json.JSONDecodeError, IndexError, KeyError) as e:
                print(f"⚠️ Error during LLM validation for claim '{claim}': {e}. Treating as unverified.")

        if not match_found:
            print(f"❌ Cache MISS for claim: '{claim}'. Adding to verification queue.")
            unverified_list.append(claim)

    state["already_verified"] = verified_list
    state["unverified_claims"] = unverified_list
    return state

# In your main agent file

def add_existing_evidence(state: AgentState) -> AgentState:
    """
    Uses source URLs from cached claims to retrieve the original evidence.
    Then, it processes this evidence against the NEW user claim.
    """
    print("-> ♻️ Re-processing evidence from cached sources...")
    cached_claims_data = state.get("already_verified", [])
    if not cached_claims_data:
        return state

    evidence_list_for_state = []

    for item in cached_claims_data:
        new_claim = item["claim"]
        source_urls = item["source_urls"]

        print(f"\n-> Re-evaluating evidence for new claim: '{new_claim}'")

        # Filter to get evidence chunks ONLY from the cached URLs
        filters = {"source_url": source_urls}
        # Retrieve chunks relevant to the NEW claim, but only from those sources
        retrieved_evidence_chunks = retrieve_chunks(new_claim, top_k=5, filters=filters)

        reprocessed_evidence = []
        if not retrieved_evidence_chunks:
            print(f"-> No relevant evidence chunks found from cached URLs for '{new_claim}'")
            continue

        # --- Re-evaluate each piece of evidence against the NEW claim ---
        embd_new_claim = embedding_model.embed_query(new_claim)
        for chunk in retrieved_evidence_chunks:
            chunk_text = chunk["text"]
            chunk_url = chunk["filterable_restricts"].get("url", [""])[0]

            # Re-calculate stance for the new claim
            stance_response = llm.invoke(stance_prompt(new_claim, chunk_text))
            stance_text = stance_response.content.strip().lower()
            stance_score = 1 if stance_text == "supports" else -1 if stance_text == "refutes" else 0

            # Re-calculate relevance for the new claim
            embd_evidence = embedding_model.embed_query(chunk_text)
            similarity = cosine_similarity([embd_new_claim], [embd_evidence])[0][0]
            relevance = (similarity + 1) / 2

            # Get credibility (this doesn't change)
            credibility = assign_credibility(chunk_url)

            reprocessed_evidence.append({
                "summary": chunk_text,  # We use the full chunk text as the "summary" here
                "title": chunk["filterable_restricts"].get("title", [""])[0],
                "url": chunk_url,
                "stance": stance_text.upper(),
                "stance_score": stance_score,
                "credibility": credibility["credibility"],
                "relevance": relevance,
                "recency": None  # You could add timestamp logic here too
            })

        # Calculate a new confidence score for the re-evaluated evidence
        if reprocessed_evidence:
            total_weight = sum(ev["stance_score"] * ev["credibility"] * ev["relevance"] for ev in reprocessed_evidence)
            confidence_score = total_weight / len(reprocessed_evidence)
        else:
            confidence_score = 0

        evidence_list_for_state.append({
            "claim": new_claim,
            "evidence": reprocessed_evidence,
            "confidence_score": confidence_score
        })

    state["text_with_evidence"] = evidence_list_for_state
    return state


# This function is now correctly defined as async
def text_evidence_collection(state: AgentState) -> AgentState:
    print("-> Starting text evidence collection...")
    claims = state["unverified_claims"]
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
                current_utc_time = datetime.now(timezone.utc)

                # 2. Format it as a standard ISO 8601 string.
                ingestion_timestamp = current_utc_time.isoformat()

                metadata = {
                    "claim": claim,
                    "ingestion_timestamp": ingestion_timestamp,
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
    existing_evidence = state.get("text_with_evidence") or []

    # Combine the cached evidence with the newly fetched evidence
    combined_evidence = existing_evidence + results_by_claim

    # Store the fully combined list back into the state
    state["text_with_evidence"] = combined_evidence
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


# --- Define the new LINEAR graph ---
graph = StateGraph(AgentState)

# Add all the nodes in sequence
graph.add_node("router", router)
graph.add_node("add_existing_evidence", add_existing_evidence)
graph.add_node("text_evidence_collection", text_evidence_collection)
graph.add_node("image_analysis", image_analysis)
graph.add_node("verify_claims", verify_claims)

# Set the entry point to the router
graph.set_entry_point("router")

# Define the linear flow of the graph
graph.add_edge("router", "add_existing_evidence")
graph.add_edge("add_existing_evidence", "text_evidence_collection")
graph.add_edge("text_evidence_collection", "image_analysis")
graph.add_edge("image_analysis", "verify_claims")
graph.add_edge("verify_claims", "__end__")

# Compile the final agent
verifier_agent = graph.compile()


def save_results_to_cache(results: dict):
    """
    Parses the final verification results and saves each newly verified claim
    to the vector database to be used as a cache.
    """
    print("-> 💾 Saving newly verified claims to cache...")
    try:
        # The result is expected to be a dictionary, not a JSON string here
        if "text_claims" not in results or not isinstance(results["text_claims"], list):
            print("-> No valid text claims found in results to cache.")
            return

        for claim_result in results["text_claims"]:
            # Embed the original claim text
            claim_text_to_embed = claim_result.get("claim")
            if not claim_text_to_embed:
                continue

            # Aggregate all source URLs into a simple list
            source_urls = [
                source.get("url") for source in claim_result.get("text_evidence_sources", []) if source.get("url")
            ]

            # Assemble the metadata for the cache entry
            metadata_to_save = {
                "doc_type": "verified_claim",
                "classification": claim_result.get("classification", "N/A"),
                "final_decision": claim_result.get("final_decision", "N/A"),
                "confidence_score": claim_result.get("confidence_score", 0.0),
                "source_urls": source_urls,
                "ingestion_timestamp": datetime.now(timezone.utc).isoformat()
            }

            print(f"-> Caching claim: '{claim_text_to_embed}'")
            # Save this verified claim back to the vector DB
            insert_into_vertex_vector_db(
                text_chunks=[claim_text_to_embed],
                metadata=metadata_to_save
            )
    except Exception as e:
        print(f"-> ⚠️ Error while saving results to cache: {e}")

# --- Tool wrapper ---
# CHANGED: The tool wrapper is now a sync function
def verifier_tool(text_news: List[str], image_path: List[str] = None, save_to_vector_db: bool = False) -> str:
    """
    Fact-checks claims using web evidence and a caching layer.
    Saves new results back to the cache if save_to_vector_db is True.
    """
    if image_path is None:
        image_path = []

    initial_state: AgentState = {
        "text_news": text_news,
        "text_with_evidence": "",
        "image_path": image_path,
        "already_verified": [],
        "unverified_claims": [],
        "image_analysis": "",
        "save_to_vector_db": save_to_vector_db,  # Pass this flag to the state
        "verified_results": ""
    }

    # Run the agent graph
    final_state = verifier_agent.invoke(initial_state)

    # --- NEW: SAVE TO CACHE LOGIC ---
    # After the agent runs, if the flag is set, save the new results.
    # We save the results from the 'verified_results' key, which contains the newly processed claims.
    if save_to_vector_db and final_state.get("verified_results"):
        # The 'verified_results' might be a dict or a JSON string depending on success
        results_to_cache = final_state["verified_results"]
        if isinstance(results_to_cache, str):
            try:
                results_to_cache = json.loads(results_to_cache)
            except json.JSONDecodeError:
                results_to_cache = {}  # Cannot parse, so cannot cache

        if isinstance(results_to_cache, dict):
            save_results_to_cache(results_to_cache)

    # --- MODIFIED: MERGE CACHED AND NEW RESULTS ---
    # The final output should combine the cached results and the new results
    final_output = {}
    cached_results = final_state.get("already_verified", [])
    new_results_dict = final_state.get("verified_results", {})

    # Ensure new_results_dict is a dictionary
    if isinstance(new_results_dict, str):
        try:
            new_results_dict = json.loads(new_results_dict)
        except json.JSONDecodeError:
            new_results_dict = {"error": "Failed to parse new results.", "raw": new_results_dict}

    final_output["cached_results"] = cached_results
    final_output["newly_verified_results"] = new_results_dict

    return json.dumps(final_output, indent=2)

# --- Test main ---
def main():
    claims = [
        "The Eiffel Tower is located in Berlin.",
        "The COVID-19 vaccine contains microchips for tracking.",
        # "Drinking water can cure cancer."
    ]
    # CHANGED: Await the async tool function
    final_results = verifier_tool(text_news=claims, image_path=[], save_to_vector_db=False)
    print("\n-> Final Verification Results:")
    print(final_results)


# CHANGED: Use asyncio.run() to execute the async main function
if __name__ == "__main__":
    main()