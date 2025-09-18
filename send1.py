import requests
import streamlit as st
import uuid
from typing import Dict, Any

# --- Page Configuration ---
st.set_page_config(
    page_title="Misinformation Classifier", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🔍 Misclassify - AI-Powered Fact Checker")
st.markdown("Enter a claim, question, or statement below to have it analyzed by our AI assistant.")

# --- API Configuration ---
CHAT_API_URL = 'http://127.0.0.1:8000/chat'

# --- Session State Initialization ---
def initialize_session_state():
    """Initialize session state variables."""
    if 'conversation_id' not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
        
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    if 'request_count' not in st.session_state:
        st.session_state.request_count = 0

def reset_conversation():
    """Reset the conversation and start fresh."""
    st.session_state.conversation_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.request_count = 0
    st.rerun()

def make_api_request(user_input: str) -> Dict[str, Any]:
    """Make API request with proper error handling."""
    payload = {
        "message": user_input,
        "conversation_id": st.session_state.conversation_id
    }
    
    response = requests.post(
        CHAT_API_URL, 
        json=payload,
        timeout=60,  # 60 second timeout
        headers={'Content-Type': 'application/json'}
    )
    response.raise_for_status()
    return response.json()

# Initialize session state
initialize_session_state()

# --- Sidebar with conversation controls ---
with st.sidebar:
    st.header("Conversation Controls")
    st.write(f"**Conversation ID:** `{st.session_state.conversation_id[:8]}...`")
    st.write(f"**Messages:** {len(st.session_state.messages)}")
    
    if st.button("🔄 Start New Conversation", type="secondary"):
        reset_conversation()
    
    if st.button("🧹 Clear Chat History", type="secondary"):
        st.session_state.messages = []
        st.rerun()

# --- Display Chat History ---
if st.session_state.messages:
    st.subheader("💬 Conversation History")
    
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
else:
    st.info("👋 Start a conversation by typing a message below!")

# --- User Input Handling ---
user_input = st.chat_input("What would you like to verify or ask about?")

if user_input:
    # Increment request counter
    st.session_state.request_count += 1
    
    # Add user message to session state and display it
    user_message = {"role": "user", "content": user_input}
    st.session_state.messages.append(user_message)
    
    with st.chat_message("user"):
        st.markdown(user_input)

    # Make API request and handle response
    try:
        with st.chat_message("assistant"):
            with st.spinner("🤖 Analyzing your message..."):
                response_data = make_api_request(user_input)
                
                ai_response = response_data.get("response", "No response received.")
                
                # Update conversation_id (should remain the same, but just in case)
                returned_conv_id = response_data.get("conversation_id")
                if returned_conv_id and returned_conv_id != st.session_state.conversation_id:
                    st.warning(f"Conversation ID changed from API: {returned_conv_id}")
                    st.session_state.conversation_id = returned_conv_id
                
                # Display the response
                st.markdown(ai_response)
                
                # Add AI response to session state
                ai_message = {"role": "assistant", "content": ai_response}
                st.session_state.messages.append(ai_message)

    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. The server might be busy. Please try again.")
        
    except requests.exceptions.ConnectionError:
        st.error("🔌 Could not connect to the API server. Please make sure the server is running on http://127.0.0.1:8000")
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 422:
            st.error("❌ Invalid request format. Please check your message.")
        elif e.response.status_code == 500:
            st.error("🚨 Server error occurred. Please try again or contact support.")
        else:
            st.error(f"❌ HTTP Error {e.response.status_code}: {e.response.text}")
            
    except requests.exceptions.RequestException as e:
        st.error(f"🔥 Request failed: {str(e)}")
        
    except Exception as e:
        st.error(f"💥 Unexpected error: {str(e)}")
        st.error("Please refresh the page and try again.")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <small>
            🔍 Misclassify AI Fact Checker | 
            Powered by LangGraph & Vertex AI | 
            <a href='http://127.0.0.1:8000/docs' target='_blank'>API Docs</a>
        </small>
    </div>
    """, 
    unsafe_allow_html=True
)