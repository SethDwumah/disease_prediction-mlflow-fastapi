import streamlit as st
from streamlit_chat import message
import requests
from helper_prabowo_ml import *

# --- FastAPI endpoint ---
API_URL = "https://disease-prediction-mlflow-fastapi.vercel.app"
# If your app is mounted under /api, switch to:
# API_URL = "https://disease-prediction-mlflow-fastapi.vercel.app/api/predict"

# --- Text cleaning helper ---
def clean_text_for_prediction(text: str) -> str:
    text = clean_html(text)
    text = remove_links(text)
    text = email_address(text)
    text = remove_digits(text)
    text = remove_special_characters(text)
    text = removeStopWords(text)
    text = punct(text)
    text = non_ascii(text)
    text = lower(text)
    return text

def init():
    st.set_page_config(page_title="Disease Prediction Assistant", page_icon="💊")
    st.header("Disease Prediction Based on Symptoms")

def main():
    init()

    # Initialize conversation history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Get user input
    user_input = st.chat_input("Describe your symptoms...", key="user_input")

    if user_input:
        # Show user message
        st.session_state.messages.append({"content": user_input, "is_user": True})

        # Clean the user input
        cleaned_text = clean_text_for_prediction(user_input)

        # Call your FastAPI
        try:
            cleaned_data = {"text": cleaned_text}
            res = requests.post(API_URL, json=cleaned_data, timeout=15)
            res.raise_for_status()
            data = res.json()

            # Extract prediction (adjust key if your API differs)
            predicted = data.get("predictions", "Unknown disease")

            reply = f"**Predicted disease:** {predicted}"

        except requests.exceptions.RequestException as e:
            # Friendly error shown in chat
            reply = f"⚠️ Error connecting to the prediction API:\n```\n{e}\n```"

        # Append response to conversation
        st.session_state.messages.append({"content": reply, "is_user": False})

    # Display the conversation
    with st.container():
        for i, msg in enumerate(st.session_state.get("messages", [])):
            message(msg["content"], is_user=msg["is_user"], key=str(i))

if __name__ == "__main__":
    main()
