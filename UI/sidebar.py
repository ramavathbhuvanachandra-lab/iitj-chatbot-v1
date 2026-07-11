import streamlit as st


def render_sidebar():

    with st.sidebar:

        st.title("🎓 IITJ Assistant")

        st.markdown("---")

        # ----------------------------
        # New Chat
        # ----------------------------

        if st.button(
            "➕ New Chat",
            use_container_width=True
        ):

            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": (
                        "👋 Hello! I am the IIT Jodhpur AI Assistant.\n\n"
                        "How can I help you today?"
                    )
                }
            ]

            st.rerun()

        st.markdown("---")

        st.subheader("About")

        st.write(
            """
This chatbot answers questions related to IIT Jodhpur.

Supported Topics:

- Admissions
- Academics
- Departments
- Research
- Hostel
- Campus Life
- Facilities
"""
        )

        st.markdown("---")

        st.caption("Built with ❤️ using")
        st.caption("• LangGraph")
        st.caption("• ChromaDB")
        st.caption("• Streamlit")