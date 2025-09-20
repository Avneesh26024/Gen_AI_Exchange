import uvicorn
import uuid
import os
import base64 
from fastapi import FastAPI, Request, HTTPException
from langchain_core.messages import HumanMessage, AIMessage

# ... other imports ...
from fastapi.middleware.cors import CORSMiddleware

from Verifier_Agent import verifier_agent
from main import agent

import asyncio
import sys

# ✅ This snippet must be here, at the top
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


app = FastAPI(
    title='Misinformation Classifier API',
    description='This API provides a conversational agent with persistent memory.'
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # allow all origins
    allow_credentials=True, # ✅ must be False if using "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR="base_images"
os.makedirs(TEMP_DIR, exist_ok=True)

# Your /chat endpoint remains the same, assuming 'agent' is also async
@app.post("/chat")
async def chat_endpoint(request: Request):
    """
    Receives a user's message and a conversation_id.
    It uses the id to retrieve the conversation's memory,
    gets a response from the agent, and returns the answer
    along with the conversation_id.
    """
    
    image_path=[]
    file_path=None
    
    try:
        data = await request.json()
        user_message = data.get("message")
        conversation_id = data.get("conversation_id") 
        image_data_base64=data.get('image_data')
        if not user_message:
            raise HTTPException(status_code=422, detail="The 'message' field is required in the JSON body.")
        
        if image_data_base64:
            try:
                image_bytes = base64.b64decode(image_data_base64)
                
                png_file = f"{uuid.uuid4()}.jpg" 
                file_path = os.path.join(TEMP_DIR, png_file)

                with open(file_path, 'wb') as out_file:
                    out_file.write(image_bytes)
                
                image_path.append(file_path)
            except Exception as e:
                print(f"Error decoding or saving base64 image: {e}")
                pass



        conversation_id = conversation_id or str(uuid.uuid4())

        config = {"configurable": {"thread_id": conversation_id}}

        agent_input = {"messages": [HumanMessage(content=user_message)],'image_path':image_path}
        final_state = agent.invoke(agent_input, config=config)

        last_ai_message = final_state["messages"][-1]
        response_content = last_ai_message.content

        return {
            "response": response_content,
            "conversation_id": conversation_id
        }

    except HTTPException as he:
        # Re-raise validation errors
        raise he
    except Exception as e:
        print(f"An error occurred: {e}")
        # Return a generic server error
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up temporary file: {file_path}")

    # ... (Your existing chat endpoint code)


@app.post('/verifier')
async def verifier_func(request: Request):
    """
    APPLIES VERIFIER AGENT
    """
    image_path = []
    file_path = None

    try:
        data = await request.json()
        user_message = data.get("message")
        conversation_id = data.get("conversation_id") 
        image_data_base64 = data.get("image_data")

        if not user_message:
            raise HTTPException(status_code=422, detail="The 'message' field is required in the JSON body.")
        
        if image_data_base64:
            try:
                image_bytes = base64.b64decode(image_data_base64)
                png_file = f"{uuid.uuid4()}.jpg"
                file_path = os.path.join(TEMP_DIR, png_file)

                with open(file_path, 'wb') as out_file:
                    out_file.write(image_bytes)

                image_path.append(file_path)
            except Exception as e:
                print(f"Error decoding or saving base64 image: {e}")
                pass

        conversation_id = conversation_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": conversation_id}}

        agent_input = {
            # ✅ FIX 2: Pass the user_message as a list to match AgentState
            "text_news": [user_message],
            "text_with_evidence": "",
            "image_path": image_path,
            "image_analysis": "",
            "save_to_vector_db": True,
            "verified_results": ""
        }
        
        # ✅ FIX 1: Use 'await' with the asynchronous 'ainvoke' method
        final_state = await verifier_agent.ainvoke(agent_input, config=config)

        response_content = final_state["verified_results"]

        return {
            "response": response_content,
            "conversation_id": conversation_id
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up temporary file: {file_path}")