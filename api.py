import uvicorn
import uuid
from fastapi import FastAPI, Request, HTTPException
from typing import List, Optional

# Import your agent and message types
from main import agent
from langchain_core.messages import HumanMessage, AIMessage

# --- Initialize the FastAPI App ---
app = FastAPI(
    title='Misinformation Classifier API',
    description='This API provides a conversational agent with persistent memory.'
)









@app.post("/chat")
async def chat_endpoint(request: Request):
    """
    Receives a user's message and a conversation_id.
    It uses the id to retrieve the conversation's memory,
    gets a response from the agent, and returns the answer
    along with the conversation_id.
    """
    try:
        # 1. Manually parse the JSON from the raw request
        data = await request.json()
        user_message = data.get("message")
        conversation_id = data.get("conversation_id") # This can be None

        # 2. Manually validate the input
        if not user_message:
            raise HTTPException(status_code=422, detail="The 'message' field is required in the JSON body.")

        # 3. Manage the Conversation ID
        # If the client doesn't provide an ID, we create a new one for a new conversation.
        conversation_id = conversation_id or str(uuid.uuid4())

        # 4. Create the LangGraph Config
        # This specific structure is required by the checkpointer to know which memory to load.\
        config = {"configurable": {"thread_id": conversation_id}}

        # 5. Invoke the Agent
        # We only need to send the latest human message. The checkpointer handles the history.
        agent_input = {"messages": [HumanMessage(content=user_message)]}
        final_state = agent.invoke(agent_input, config=config)

        # 6. Extract the Final Response
        # The agent's final state contains the full history; we just need the last AI message.
        last_ai_message = final_state["messages"][-1]
        response_content = last_ai_message.content

        # 7. Return the Response and ID as a simple dictionary
        return {
            "response": response_content,
            "conversation_id": conversation_id
        }

    except HTTPException as he:
        # Re-raise validation errors
        raise he
    except Exception as e:
        # It's good practice to log the error properly in a real application
        print(f"An error occurred: {e}")
        # Return a generic server error
        raise HTTPException(status_code=500, detail=str(e))

# if name == "main":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


