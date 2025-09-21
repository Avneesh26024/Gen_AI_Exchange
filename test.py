from logging import lastResort
from nodes import verifier_tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory, ChatVertexAI
from prompts import main_prompt
from typing import TypedDict, List, Any, Dict
from Retriever.Retriever_Agent import retriever_agent
from Tools.Human_Response import human_response
from langchain.prompts import ChatPromptTemplate

# --- Model Configuration (Unchanged) ---
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
model = ChatVertexAI(
    model_name="gemini-2.5-flash-lite",
    **model_kwargs
)


class AgentState(TypedDict):
    verified_claims: List[Dict[str, Any]]
    messages: List[str]

def router(state: AgentState) -> StateGraph:

    return state

def relevant_context_router(state: AgentState) -> AgentState:
    verified_claims = state.get("verified_claims", [])
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else ""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant which is a part of the misinformation classifier. You will be given a user message, chat history and a list of previously verified claims given by user. Your task is to determine if the latest user message can be answered using the  "
    ])

def check_claims_router(state: AgentState) -> AgentState:
    verified_claims = state.get("verified_claims", [])
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else ""

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You will be given a user message and a list of previously verified claims. "
                   "Determine if the user message is a new claim that needs verification or if it has already been verified. "
                   "If it is a new claim, respond with 'new_claim'. If it has already been verified, respond with 'already_verified'."),
        ("user", "Previously verified claims: {verified_claims}"),
        ("user", "User message: {last_message}"),
    ])

    chain = prompt | model
    response = chain.invoke({
        "verified_claims": verified_claims,
        "last_message": last_message
    })

    # Normalize response (depends on model type)
    decision = response.content.strip().lower()

    return decision

def relevant_context(state: AgentState) -> AgentState:
    # Extract from state
    verified_claims = state.get("verified_claims", [])
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else ""

    # Build the prompt for query generation
    retrieve_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an intelligent evidence retriever agent. "
            "Your task is to generate a clear and specific query for the retriever_agent, "
            "which searches a vector database and returns relevant evidence chunks. "
            "You are part of a misinformation classification system and you already know "
            "which claims have been verified. "
            "Given the list of verified claims, the latest user message, and the chat history, "
            "produce ONLY the query text—no explanations, commentary, or formatting. "
            "The query must be self-contained and directly usable by another agent."
        ),
        (
            "user",
            "Previously verified claims:\n{verified_claims}"
        ),
        (
            "user",
            "Latest user message:\n{last_message}"
        ),
        (
            "user",
            "Chat history:\n{messages}"
        ),
    ])

    # Run the LLM chain to generate the retrieval query
    chain = retrieve_prompt | model
    response = chain.invoke({
        "verified_claims": verified_claims,
        "last_message": last_message,
        "messages": messages,
    })

    # Extract the query from model output
    query = response.content.strip()

    # Pass the query to retriever agent
    chunks = retriever_agent(query=query)

    # Format retrieved chunks into a single string
    formatted = ""
    for chunk in chunks:
        formatted += f"Retrieved Chunk Content: {chunk.get('text', '')}\n"
        formatted += (
            "Additional Information Related to the Chunk: "
            f"{chunk.get('full_metadata', '')}, "
            f"{chunk.get('filterable_restricts', '')}, "
            f"{chunk.get('numeric_restricts', '')}\n"
        )

    # Store in state
    state["relevant_context"] = formatted
    return state

def classify_claim(state: AgentState) -> AgentState:
    verified_claims = state.get("verified_claims", [])
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else ""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI agent which is a part of the misinformation classification system. You will be given the list of claims which have been verified till now. You will be given conversation history and the latest user message. You have to extract the claim/claims from the last user message if any. The output should be in a list format. If there is no claim in the last user message, return an empty list. Separate each claim with a #. Do not include any other text."),
        ("user", "Previously verified claims: {verified_claims}"),
        ("user", "Conversation history: {messages}"),
        ("user", "Latest user message: {last_message}"),
        ])
    chain = prompt | model
    response = chain.invoke({
        "verified_claims": verified_claims,
        "messages": messages,
        "last_message": last_message
    })
    claims_text = response.content.strip()
    claims = [claim.strip() for claim in claims_text.split("#") if claim.strip()]
    classification = verifier_tool.invoke(claims=claims, save_to_vector_db=True)
    state["verified_claims"].extend(classification)

