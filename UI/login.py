import streamlit as st

from backend.user_db import get_or_create_user
from backend.session_db import create_session


def render_login():

    st.title("🎓 IIT Jodhpur AI Assistant")

    st.subheader("Welcome!")

    st.write("Please enter your details to continue.")

    name = st.text_input("Full Name")

    phone = st.text_input(
        "Phone Number",
        max_chars=10,
        placeholder="Enter 10-digit mobile number"
    )



    if st.button("Continue", use_container_width=True):

        if len(name.strip()) < 3:
            st.error("Please enter your full name.")
            return
        
        phone = phone.strip()
        
        if not phone.strip() or len(phone) != 10:
            st.error("Please enter your phone number.")
            return

        user = get_or_create_user(name, phone)

        session = create_session(user["user_id"])

        st.session_state.logged_in = True
        st.session_state.user_id = user["user_id"]
        st.session_state.user_name = user["name"]
        st.session_state.session_id = session["session_id"]

        st.rerun()