import streamlit as st


# ---------------------------------------------------------
# Initialize Session State
# ---------------------------------------------------------

def initialize_chat():

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 Hello! I am the IIT Jodhpur AI Assistant.\n\n"
                    "Ask me anything about:\n"
                    "• Admissions\n"
                    "• Academics\n"
                    "• Departments\n"
                    "• Research\n"
                    "• Hostel\n"
                    "• Campus Facilities"
                )
            }
        ]


# ---------------------------------------------------------
# Display Previous Chat
# ---------------------------------------------------------

def display_chat_history():

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])


# ---------------------------------------------------------
# Backend Response
# ---------------------------------------------------------

# ---------------------------------------------------------
# Backend Response
# ---------------------------------------------------------

def get_response(question, graph):

    try:

        # Last 10 messages (5 user-assistant exchanges)
        chat_history = st.session_state.messages[-10:]

        result = graph.invoke(
            {
                "question": question,
                "chat_history": chat_history
            }
        )

        return result["answer"]

    except Exception as e:

        return f"❌ Error:\n\n{e}"

# ---------------------------------------------------------
# Handle User Prompt
# ---------------------------------------------------------

def handle_user_prompt(prompt, graph):

    # Store User Message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    # Display User Message

    with st.chat_message("user"):

        st.markdown(prompt)

    # Assistant Response

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer = get_response(
                question=prompt,
                graph=graph
            )

            st.markdown(answer)

    # Save Assistant Message

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )