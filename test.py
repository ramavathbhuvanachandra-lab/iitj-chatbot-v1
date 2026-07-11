import streamlit as st

st.set_page_config(
    page_title="Test",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.title("Sidebar Test")
    st.button("New Chat")

st.title("Hello World")