import streamlit as st

# ---------------------------------------------------
# Page Configuration (MUST BE FIRST)
# ---------------------------------------------------

st.set_page_config(
    page_title="IITJ Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------
# Imports
# ---------------------------------------------------

from backend.graph import create_graph
from UI.sidebar import render_sidebar
from UI.chat import (
    initialize_chat,
    display_chat_history,
    handle_user_prompt,
)

# ---------------------------------------------------
# Backend Graph (Load Once)
# ---------------------------------------------------

graph = create_graph()

# ---------------------------------------------------
# Session State
# ---------------------------------------------------

initialize_chat()

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

render_sidebar()

# ---------------------------------------------------
# Main Page
# ---------------------------------------------------

st.title("🎓 IIT Jodhpur AI Assistant")

st.caption(
    "Ask anything about IIT Jodhpur • Admissions • Academics • Research • Hostel • Campus"
)

# ---------------------------------------------------
# Chat History
# ---------------------------------------------------

display_chat_history()

# ---------------------------------------------------
# User Input
# ---------------------------------------------------

prompt = st.chat_input(
    "Ask anything about IIT Jodhpur..."
)

# ---------------------------------------------------
# Generate Response
# ---------------------------------------------------

if prompt:
    handle_user_prompt(
        prompt=prompt,
        graph=graph
    )