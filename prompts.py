from typing import List
import json
# --- Schema Templates (Helper constants to keep the main function clean) ---

TEXT_CLAIMS_SCHEMA = """
  "text_claims": [
    {
      "claim": "The original text claim being verified.",
      "classification": "REAL | FAKE | REAL with EDGE CASE",
      "edge_case_notes": "A brief explanation if the classification is 'REAL with EDGE CASE'. Otherwise, this should be an empty string.",
      "text_evidence_summary": "A 1-2 sentence summary of the web evidence for the text claim.",
      "text_evidence_sources": [
        {
          "url": "Source URL",
          "title": "Source Title",
          "stance": "SUPPORTS | REFUTES | NO STANCE",
          "credibility": 0.9
        }
      ],
      "image_correlation": "Supports (Image X) | Contradicts (Image X) | No relevant evidence",
      "confidence_score": -1.0,
      "final_decision": "A final 1-2 sentence verdict on the claim."
    }
  ]
"""

IMAGES_SCHEMA = """
  "images": [
    {
      "image_id": 1,
      "image_path": "path/to/image.jpg",
      "ocr_text": "Text found in the image, if any.",
      "labels": "Labels identified in the image.",
      "caption": "AI-generated caption for the image.",
      "image_evidence_summary": "A 1-2 sentence summary of the web evidence found for this image.",
      "text_evidence_sources": [
        {
          "url": "Source URL",
          "title": "Source Title",
          "stance": "SUPPORTS | REFUTES | NO STANCE",
          "credibility": 0.9
        }
      ],
      "confidence_score": 0.9,
      "final_decision": "A final 1-2 sentence conclusion about the image's context."
    }
  ]
"""


def verifier_prompt(state: dict) -> str:
    """
    Dynamically generates the verifier prompt based on whether text claims,
    images, or both are provided in the agent's state.
    """
    # --- 1. Check for the presence of text and image data ---
    has_text = bool(state.get("text_with_evidence"))
    has_images = bool(state.get("image_analysis"))

    # --- 2. Dynamically build the sections of the prompt ---
    input_sections = []
    schema_sections = []
    main_instructions = [
        "You are an expert, impartial Fact-Checking AI Agent.",
        "Your sole task is to synthesize the provided evidence into a structured JSON output.",
        "Your entire response MUST be a single, valid JSON object without any markdown formatting (like ```json).",
        "Be concise and objective in all summary and decision fields."
    ]

    # --- Case 1: Both Text and Images are provided ---
    if has_text and has_images:
        main_instructions.append(
            "Analyze both text and image evidence. For each text claim, determine if any of the provided images correlate. "
            "If an image is not relevant to any claim, you MUST state 'No relevant evidence' in the claim's `image_correlation` field."
        )
        # Input sections
        text_evidence_str = json.dumps(state["text_with_evidence"], indent=2)
        input_sections.append(f"#### 1. Text Claims and Collected Web Evidence:\n```json\n{text_evidence_str}\n```")
        input_sections.append(f"#### 2. Image Analysis and Collected Web Evidence:\n```\n{state['image_analysis']}\n```")
        # Schema sections
        schema_sections.append(TEXT_CLAIMS_SCHEMA)
        schema_sections.append(IMAGES_SCHEMA)

    # --- Case 2: Only Text is provided ---
    elif has_text and not has_images:
        main_instructions.append(
            "Only text claims were provided. Analyze the evidence for each claim and populate the `text_claims` section. "
            "Since no images were provided, `image_correlation` MUST be 'No relevant evidence' for all claims and the `images` list MUST be empty."
        )
        # Input section
        text_evidence_str = json.dumps(state["text_with_evidence"], indent=2)
        input_sections.append(f"#### Text Claims and Collected Web Evidence:\n```json\n{text_evidence_str}\n```")
        # Schema sections
        schema_sections.append(TEXT_CLAIMS_SCHEMA)
        schema_sections.append('"images": []') # Explicitly require an empty list

    # --- Case 3: Only Images are provided ---
    elif not has_text and has_images:
        main_instructions.append(
            "Only images were provided. Analyze the evidence for each image and populate the `images` section. "
            "Since no text claims were provided, the `text_claims` list MUST be empty."
        )
        # Input section
        input_sections.append(f"#### Image Analysis and Collected Web Evidence:\n```\n{state['image_analysis']}\n```")
        # Schema sections
        schema_sections.append('"text_claims": []') # Explicitly require an empty list
        schema_sections.append(IMAGES_SCHEMA)

    # --- Fallback Case: No input provided ---
    else:
        return '{"text_claims": [], "images": []}' # Return a valid empty JSON

    # --- 3. Assemble the final prompt from the dynamic sections ---
    final_instructions = "\n".join(f"- {inst}" for inst in main_instructions)
    final_inputs = "\n\n".join(input_sections)
    final_schema = ",\n".join(schema_sections)

    return f"""
### INSTRUCTIONS
{final_instructions}

---
### EVIDENCE INPUT
{final_inputs}

---
### REQUIRED JSON OUTPUT SCHEMA
Your final output must conform exactly to this JSON structure.

```json
{{
{final_schema}
}}
"""

def content_summarizer_prompt(content: str) -> str:
    return f"""
    You are a fact-preserving summarizer.

    ### Input
    Content:
    {content}

    ### Task
    Summarize the content into a shorter version that is **concise but does not omit any
    important factual details, numbers, names, or claims**.
    - Remove filler text, repetition, and irrelevant context.
    - Keep all essential information that could be useful for fact-checking later.
    - The result should be significantly shorter than the original, but still detailed enough 
      to capture every necessary fact.
    """


def retriever_prompt() -> List[str]:
    """
    Returns system instructions for the retriever agent specifically
    for retrieving evidence chunks related to claim verification.

    Instructions include how to handle the available metadata fields
    for filtering retrieved chunks.
    """
    return [
        "You are an intelligent evidence retriever agent specialized in claim verification.",
        "Your goal is to retrieve the most relevant chunks from a vector database given a user query (a claim).",
        "You can filter the chunks based on the following metadata fields:",
        # "  1. user_id: unique identifier of the user who initiated the verification",  # (Uncomment if needed later)
        "  1. claim: the specific claim the evidence is linked to",
        "  2. source_url: the URL of the original source document/webpage",
        "  3. title: the title of the source document/webpage",
        "When retrieving chunks, consider both the user query and any explicit filters provided.",
        "Always return the retrieved chunks in a structured JSON format like:",
        "{'retrieved_chunks': [{'id': '...', 'text': '...', 'metadata': {...}}, ...]}",
        "Do not include commentary or unrelated text.",
        "Ensure that the output is complete and can be parsed by another agent or tool."
    ]


from langchain_core.prompts import ChatPromptTemplate

def main_prompt() -> ChatPromptTemplate:
    """
    Creates the prompt for the ReAct agent, which is now specialized
    for verifying new claims.
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a specialized **AI Fact-Checker**.
Your sole purpose is to verify new, unverified claims presented by a user. You have been activated because the initial query was identified as a new claim requiring verification.

### Your Workflow:
Your goal is to use your tools to get a definitive classification for the user's claim.

1.  **Assess the Claim**: Analyze the user's latest message, which contains the claim to be verified.
2.  **Verify or Clarify**:
    * If the claim is clear and specific, your primary action is to use the `verifier_tool` to get a final classification.
    * If the claim is ambiguous, vague, or missing crucial information (like a link for an article), you **must** use the `human_response` tool to ask the user for the necessary details.

### Your Tools:
- `verifier_tool`: The only tool that can provide a final classification of a claim.
- `human_response`: Use this to ask the user for clarification.

### Core Directives:
- **Your ONLY job is to verify this specific claim.** Do not engage in general conversation.
- **NEVER classify a claim yourself.** Your final answer must come from the output of the `verifier_tool`.
- **Do not assume context.** If you need more information, ask the user.
""",
            ),
            # The 'messages' placeholder will be filled by the ReAct agent with the conversation history.
            ("user", "{messages}"),
        ]
    )

def stance_prompt(claim: str, evidence: str) -> str:
    return f"""
You are an expert Stance Detection AI Agent.

Claim:
{claim}

Evidence:
{evidence}

Task:
Determine the stance of the evidence with respect to the claim.

Output:
- SUPPORTS → evidence clearly supports the claim
- REFUTES → evidence clearly contradicts the claim
- NO STANCE → evidence is neutral or unrelated

Return ONLY one of: SUPPORTS, REFUTES, NO STANCE. No extra text.
"""

# In prompts.py

def validation_prompt(claim: str, retrieved_claims: list) -> str:
    """
    Creates a prompt to ask an LLM if a new claim is semantically identical
    to any of the retrieved, already-verified claims.
    """
    # Format the retrieved claims for clear presentation in the prompt
    retrieved_claims_str = "\n".join(
        f'- ID {i+1}: "{c["text"]}" (Final Decision: "{c["filterable_restricts"].get("final_decision", ["N/A"])[0]}")'
        for i, c in enumerate(retrieved_claims)
    )

    return f"""
You are an AI validation assistant. Your task is to determine if the "New Claim" is semantically identical to any of the "Previously Verified Claims" found in our database. Semantically identical means they ask the same question or state the same fact, even if the wording is slightly different.

### New Claim:
"{claim}"

### Previously Verified Claims:
{retrieved_claims_str}

### INSTRUCTIONS:
1.  Carefully compare the "New Claim" to each of the "Previously Verified Claims".
2.  If you find a claim that is a perfect semantic match, identify its ID.
3.  Your response MUST be a single JSON object with the following structure:
    {{
      "match_found": boolean,
      "matched_id": integer | null
    }}
4.  Set "match_found" to `true` if an identical claim exists, otherwise `false`.
5.  If "match_found" is `true`, set "matched_id" to the corresponding integer ID (e.g., 1, 2, ...). Otherwise, set it to `null`.
6.  Do not add any explanations or extra text outside of the JSON object.

### Example Response:
If Previously Verified Claim with ID 2 was a perfect match, you would return:
```json
{{
  "match_found": true,
  "matched_id": 2
}}
"""