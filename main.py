from Verifier_Agent import verifier_tool
from typing import TypedDict, List
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory, ChatVertexAI


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

llm = ChatVertexAI(
    model_name="gemini-2.5-flash-lite", # Using a stable model name
    **model_kwargs
)


class AgentState(TypedDict):
    chat_history: List[str]
    images: List[str]

def checker_prompt(chat: str, query: str)-> str:
    return f"""
    You are an AI agent which is a part of a misinformation classifier agent.
    Your task is to determine if the user's query can be answered with the provided chat history.
    In the chat history you will see the claims made by the user and the responses given by the AI.
    You will be given a user query and the chat history.
    If the chat history contains information that can be used to answer the user's query, respond with "yes".
    If the user's query is related to a already verified claim, respond with "RETRIEVE".
    
    Here's the chat history:
    {chat}
    
    Here's the user's query:
    {query}
    
    
"""



#CHECKER NODE
def checker(state: AgentState)-> AgentState:
    """Checks if the user's query can be answered with chat_history."""
    #First check if the user last prompt had any images or not
    num_img = len(state.get("images", []))
    if num_img > 0:
        #Here it has already been decided that the further process will be transferred to the verifier agent.
        return "VERIFY"

    messages = state.get("chat_history", [])
    user_query = messages[-1] if messages else ""
    chat_history = "\n".join(messages[:-1]) if len(messages) > 1 else ""
    prompt = checker_prompt(chat_history, user_query)
    response = llm.invoke(prompt)
    next = (response.content()).strip().lower()
    if next == "yes":
        return "FINAL_ANSWER"
    elif next == "retrieve":
        return "VERIFY"
    else:
        return "RETRIEVE"




