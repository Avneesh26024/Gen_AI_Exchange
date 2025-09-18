import uvicorn
import uuid
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

# Import your agent and message types
from main1 import agent
from langchain_core.messages import HumanMessage, AIMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Request/Response Models ---
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str

# --- Initialize the FastAPI App ---
app = FastAPI(
    title='Misinformation Classifier API',
    description='This API provides a conversational agent with persistent memory.',
    version="1.0.0"
)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    """
    Receives a user's message and a conversation_id.
    It uses the id to retrieve the conversation's memory,
    gets a response from the agent, and returns the answer
    along with the conversation_id.
    """
    try:
        user_message = chat_request.message.strip()
        conversation_id = chat_request.conversation_id

        # Validate input
        if not user_message:
            raise HTTPException(
                status_code=422, 
                detail="The 'message' field cannot be empty."
            )

        # Generate conversation ID if not provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            logger.info(f"Generated new conversation ID: {conversation_id}")
        else:
            logger.info(f"Using existing conversation ID: {conversation_id}")

        # Create the LangGraph Config
        config = {"configurable": {"thread_id": conversation_id}}

        # Prepare agent input - only send the new human message
        # The memory system will handle conversation history
        agent_input = {
            "messages": [HumanMessage(content=user_message)],
            "verified_results": "",
            "relevant_context": "",
            "condensed_query": "",
            "image_path": []
        }

        logger.info(f"Processing message: {user_message[:100]}...")

        # Invoke the Agent
        final_state = agent.invoke(agent_input, config=config)

        # Extract the final response
        if not final_state.get("messages"):
            raise HTTPException(
                status_code=500,
                detail="Agent did not produce any response"
            )

        # Get the last AI message
        last_ai_message = None
        for message in reversed(final_state["messages"]):
            if isinstance(message, AIMessage):
                last_ai_message = message
                break

        if not last_ai_message:
            raise HTTPException(
                status_code=500,
                detail="Agent did not produce an AI response"
            )

        response_content = last_ai_message.content
        logger.info(f"Generated response: {response_content[:100]}...")

        return ChatResponse(
            response=response_content,
            conversation_id=conversation_id
        )

    except HTTPException as he:
        logger.error(f"HTTP Exception: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Misinformation Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )