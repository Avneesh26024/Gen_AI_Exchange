import requests
import streamlit as st

# --- Page Configuration ---
st.set_page_config(page_title="Misinformation Classifier", layout="wide")
st.title("Welcome to Misclassify")
st.markdown("Enter a claim, question, or statement below to have it analyzed.")

# --- API Configuration ---
CHAT_API_URL = 'http://127.0.0.1:8000/chat'


if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = "14878c64-6af8-4790-a734-b5059764f970"
if 'messages' not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("What would you like to verify?")

if user_input:
   
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    payload = {
        "message": user_input,
        "conversation_id": st.session_state.conversation_id
    }

    try:
        with st.spinner("Analyzing..."):
            response = requests.post(CHAT_API_URL, json=payload)
            response.raise_for_status()
            response_data = response.json()

            ai_response = response_data.get("response")

            st.session_state.conversation_id = response_data.get("conversation_id")

            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            with st.chat_message("assistant"):
                st.markdown(ai_response)

    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while communicating with the agent: {e}")

