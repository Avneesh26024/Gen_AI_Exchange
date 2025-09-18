# First Prototype the Vertex AI Search Here
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from Verifier_Agent import verifier_tool
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
    "max_output_tokens": 1000,
    "top_p": 0.95,
    "top_k": None,
    "safety_settings": safety_settings,
}
model = ChatVertexAI(
    model_name="gemini-2.5-flash-lite",
    **model_kwargs
)
Tools = [verifier_tool]

agent = create_react_agent(
    model=model,
    tools=Tools,
    prompt = SystemMessage(content="You are an AI assistant that helps users verify factual claims using the verifier_agent tool.")
)
response = agent.invoke({"input": "Is it true that the Eiffel Tower is located in Berlin?"})
print("Final Response:", response["output"])
