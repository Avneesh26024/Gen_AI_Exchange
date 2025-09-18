import redis
import json
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from Utils.vector_db import insert_into_vertex_vector_db
from prompts import redis_prompt

"""Basic connection example.
"""

import redis

r = redis.Redis(
    host='redis-13432.c52.us-east-1-4.ec2.redns.redis-cloud.com',
    port=13432,
    decode_responses=True,
    username="default",
    password="06CXltZ8y1dyvK2WeZzRUeWFcymsltcA",
)

llm=ChatOllama(base_url='http://localhost:11434',model='llama3')



success = r.set('foo', 'bar')
# True

result = r.get('foo')


MESSAGE_LIMIT=50


def add_message(user, message):
    """Add message to Redis (cache)."""
    r.rpush("chat_history", f"{user}: {message}")

    # if above threshold → summarize & store into FAISS
    if r.llen("chat_history") > MESSAGE_LIMIT:
        flush_to_fiass()




def flush_to_fiass():
    """Take messages from Redis, join them, push to FAISS, clear cache."""
    history = r.lrange("chat_history", 0, -1)
    if not history:
        return
    
    ### yha pe llm hoga 
    prompt=redis_prompt(history)


    


    
    
    

    # Simple summary = just join for demo (replace with LLM summary if needed)
    faiss.add_texts(history)


    # Add to FAISS


    # Save FAISS
    faiss.save_local(vectorstore_path)

    # Clear Redis
    r.delete("chat_history")

    print("✅ Flushed Redis -> FAISS")
