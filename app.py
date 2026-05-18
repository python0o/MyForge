import streamlit as st
from dotenv import load_dotenv
import os
import requests
from graph import create_myforge_graph
from state import MyForgeState

load_dotenv()

st.set_page_config(page_title="MyForge", layout="wide")
st.title("🔨 MyForge")

# Sidebar Settings
st.sidebar.header("Settings")
domain = st.sidebar.selectbox("Domain", ["Framing", "Plumbing", "Electrical", "Flooring", "HVAC"])

voice_options = {
    "Framing": {"Calm Guide": "21m00Tcm4TlvDq8ikWAM"},
    "Plumbing": {"Practical": "ErXwobaYiN019PkySvjV"},
    "Electrical": {"Clear Instructor": "AZnzlk1XvdvUeBnXmlld"},
    "Flooring": {"Steady Voice": "TxGEqnHWrfWFTfGW9XjX"},
    "HVAC": {"Technical": "pNInz6obpgDQGcFmaJgB"}
}
selected_voice = st.sidebar.selectbox("Voice", list(voice_options[domain].keys()))
voice_id = voice_options[domain][selected_voice]

# Session memory
if "history" not in st.session_state:
    st.session_state.history = []
if "last_response" not in st.session_state:
    st.session_state.last_response = ""

# Domain-specific graph (for now using main graph)
graph = create_myforge_graph()

problem = st.text_area(f"Describe your {domain.lower()} issue:")

if st.button("Get Guidance"):
    state: MyForgeState = {
        "conversation_history": st.session_state.history,
        "problem_description": problem,
        "current_diagnosis": None,
        "last_step": None,
        "awaiting_confirmation": False,
        "confirmed_steps": [],
        "mode": "human",
        "domain": domain.lower(),
        "selected_voice_id": voice_id
    }
    result = graph.invoke(state)
    response = result.get("last_step", "No guidance generated.")
    st.session_state.history.append(f"You: {problem}")
    st.session_state.history.append(f"MyForge: {response}")
    st.session_state.last_response = response

# Show conversation
for msg in st.session_state.history:
    st.write(msg)

# Confirmation buttons
if st.session_state.get("awaiting_confirmation"):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes"):
            st.session_state.history.append("You: Yes, proceed")
            st.session_state.awaiting_confirmation = False
            st.rerun()
    with col2:
        if st.button("No"):
            st.session_state.history.append("You: No, stop")
            st.session_state.awaiting_confirmation = False
            st.rerun()

# Voice Output
if st.button("🔊 Speak Response"):
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if api_key and st.session_state.last_response:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {"xi-api-key": api_key}
        data = {"text": st.session_state.last_response}
        r = requests.post(url, json=data, headers=headers)
        if r.status_code == 200:
            st.audio(r.content, format="audio/mp3")
        else:
            st.error("Voice generation failed.")
    else:
        st.warning("ElevenLabs key missing or no response to speak.")