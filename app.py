import streamlit as st
from UI.login import render_login
import time
from UI.sidebar import render_sidebar
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

from UI.chat import (
    initialize_chat,
    display_chat_history,
    handle_user_prompt,
)
from UI.login import render_login
from UI.components import quick_action_cards
# ---------------------------------------------------
# Backend Graph (Load Once)
# ---------------------------------------------------

graph = create_graph()

# ---------------------------------------------------
# Session State
# ---------------------------------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    render_login()
    st.stop()

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
# Quick Action Cards
# ---------------------------------------------------

active_chat = st.session_state.active_chat

messages = st.session_state.conversations[active_chat]["messages"]

# Show only before the first user message
if len(messages) == 1:

    selected_question = quick_action_cards()

    if selected_question:

        handle_user_prompt(
            prompt=selected_question,
            graph=graph,
        )

        st.rerun()

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