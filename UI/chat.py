import streamlit as st
import uuid
import time 
from backend.message_db import save_message
from backend.assistant_router import assistant_router
import traceback

# ---------------------------------------------------------
# Initialize Session State
# ---------------------------------------------------------

def initialize_chat():

    if "conversations" not in st.session_state:

        chat_id = str(uuid.uuid4())

        st.session_state.conversations = {
            chat_id: {
                "title": "New Chat",
                "messages": [
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
            }
        }

        st.session_state.active_chat = chat_id



# ---------------------------------------------------------
# Create New Chat
# ---------------------------------------------------------

def create_new_chat():

    chat_id = str(uuid.uuid4())

    st.session_state.conversations[chat_id] = {
        "title": "New Chat",
        "messages": [
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
    }



    st.session_state.active_chat = chat_id
# -
# ---------------------------------------------------------
# Generate Chat Title
# ---------------------------------------------------------

def generate_chat_title(prompt: str) -> str:
    """
    Generate a clean chat title from the user's first message.
    """

    title = prompt.strip()

    # Remove common starting phrases
    prefixes = [
        "tell me about",
        "can you tell me about",
        "what is",
        "what are",
        "give me",
        "explain",
        "explain about",
        "information about",
    ]

    lower_title = title.lower()

    for prefix in prefixes:

        if lower_title.startswith(prefix):

            title = title[len(prefix):].strip()

            break

    # Remove punctuation
    title = title.strip(" ?!.,:")

    # Capitalize nicely
    title = title.title()

    # Limit length
    if len(title) > 30:
        title = title[:30] + "..."

    # Fallback
    if title == "":
        title = "New Chat"

    return title 


# ---------------------------------------------------------
# Format Chat History
# ---------------------------------------------------------

def format_chat_history(messages):
    """
    Convert conversation into a readable format for the LLM.

    - Removes the welcome message.
    - Excludes the current user message.
    """

    history = []

    for message in messages:

        # Skip welcome message
        if (
            message["role"] == "assistant"
            and "Hello! I am the IIT Jodhpur AI Assistant" in message["content"]
        ):
            continue

        role = message["role"].capitalize()

        history.append(
            f"{role}:\n{message['content']}"
        )

    return "\n\n".join(history)
#--------------------------------------------------------
# Display Previous Chat
# ---------------------------------------------------------



def display_chat_history():

    active_chat = st.session_state.active_chat

    messages = st.session_state.conversations[active_chat]["messages"]

    for message in messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])
            
            if (
                message["role"] == "assistant"
                and "response_time" in message
            ):
                st.caption(
                    f"⏱️ Response Time: {message['response_time']:.2f} seconds"
                )

# ---------------------------------------------------------
# Backend Response
# ---------------------------------------------------------

# ---------------------------------------------------------
# Backend Response
# ---------------------------------------------------------

# ---------------------------------------------------------
# Backend Response
# ---------------------------------------------------------

def get_response(question, graph):

    try:

        active_chat = st.session_state.active_chat
        messages = st.session_state.conversations[active_chat]["messages"]

        # Exclude current user question
        previous_messages = messages[:-1]

        # Last 10 previous messages
        previous_messages = previous_messages[-10:]

        chat_history = format_chat_history(previous_messages)

        # -------------------------------
        # Structured Routing
        # -------------------------------
        route = assistant_router(question)

        final_response = ""

        # Navigation / Emergency response
        for item in route["structured"]:

            if item["type"] == "navigation":

                location = item["data"]
                final_response += f"📍 **{location['name']}**\n\n"
                if location.get("description"):
                    final_response += f"{location['description']}\n\n"

                if location.get("timings"):
                    final_response += f"🕒 **Timings:** {location['timings']}\n\n"
                if location.get("google_maps"):
                    final_response += (
                        f"🗺️ **Google Maps:**\n"
                        f"{location['google_maps']}\n\n"
                    )

            elif item["type"] == "emergency":

                 contact = item.get("data")

                 if contact:
                      if contact.get("name"):
                       final_response += f"📍 **{contact['name']}**\n\n"

                        # Description
                      if contact.get("description"):
                       final_response += f"{contact['description']}\n\n"

                       # Timings
                       if contact.get("timings"):
                           final_response += f"🕒 **Timings:** {contact['timings']}\n\n"

                        # Google Maps
                       if contact.get("google_maps"):
                           final_response += (
                              f"🗺️ **Google Maps:**\n"
                              f"{contact['google_maps']}\n\n"
                            )
      
              

                

        total_time = 0

        # -------------------------------
        # Existing RAG Workflow
        # -------------------------------
        if route["need_rag"]:

            start = time.time()
            try:

               result = graph.invoke(
                 {
                    "question": question,
                    "chat_history": chat_history,
                 }
                )
            except Exception:
                traceback.print_exc()
                raise   


            end = time.time()
            total_time = end - start

            print("\n" + "=" * 80)
            print(f"🚀 TOTAL WORKFLOW TIME : {total_time:.2f} sec")
            print("STREAMLIT GRAPH RESULT")
            print("=" * 80)

            for key, value in result.items():
                print(f"\n----- {key} -----")
                print(value)

            print("=" * 80)

            if final_response:
                final_response += "\n---\n\n"

            final_response += result["answer"]

        return final_response, total_time

    except Exception as e:
        print("=" * 80)
        traceback.print_exc()
        print("=" * 80)

        return f"❌ Error:\n\n{e}", 0
# ---------------------------------------------------------
# Handle User Prompt
# ---------------------------------------------------------
def handle_user_prompt(prompt, graph):

    # Get Active Conversation
    active_chat = st.session_state.active_chat
    messages = st.session_state.conversations[active_chat]["messages"]

    # ---------------- USER MESSAGE ---------------- #

    messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    try:
        save_message(
            session_id=st.session_state.session_id,
            role="user",
            message=prompt
        )
    except Exception as e:
        print("=" * 80)
        traceback.print_exc()
        print("=" * 80)
        print(f"Failed to save user message: {e}")

    # Set Chat Title
    if st.session_state.conversations[active_chat]["title"] == "New Chat":
        st.session_state.conversations[active_chat]["title"] = (
            generate_chat_title(prompt)
        )

    with st.chat_message("user"):
        st.markdown(prompt)

    # ---------------- ASSISTANT RESPONSE ---------------- #

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer, response_time = get_response(
                question=prompt,
                graph=graph
            )

        # Display Answer
        st.markdown(answer)
        st.caption(f"⏱️ Response Time: {response_time:.2f} seconds")

        # Save assistant message in memory
        messages.append(
            {
                "role": "assistant",
                "content": answer,
                "response_time": response_time
            }
        )

        # Save assistant message in database
        try:
            saved_message = save_message(
                session_id=st.session_state.session_id,
                role="assistant",
                message=answer,
                response_time=response_time
            )

            # Save database message id
            messages[-1]["message_id"] = saved_message["id"]

        except Exception as e:
            print(f"Failed to save assistant message: {e}")
            messages[-1]["message_id"] = None

        