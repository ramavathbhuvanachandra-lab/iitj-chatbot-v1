import streamlit as st
import uuid
import time 


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
        start = time.time()
        result = graph.invoke(
            {
                "question": question,
                "chat_history": chat_history,
            }
        )
        end = time.time()
        print("\n" + "=" * 80)
        print(f"🚀 TOTAL WORKFLOW TIME : {end - start:.2f} sec")
        print("STREAMLIT GRAPH RESULT")
        print("=" * 80)
        
        for key, value in result.items():
            print(f"\n----- {key} -----")
            print(value)

        print("=" * 80)


        

        return result["answer"],end - start
        

    except Exception as e:

        return f"❌ Error:\n\n{e}",0
# ---------------------------------------------------------
# Handle User Prompt
# ---------------------------------------------------------

def handle_user_prompt(prompt, graph):

    # Get Active Conversation
    active_chat = st.session_state.active_chat
    messages = st.session_state.conversations[active_chat]["messages"]

    # Store User Message
    messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )
    # Display User Message
    # Set Chat Title (First User Message)
    if st.session_state.conversations[active_chat]["title"] == "New Chat":
         st.session_state.conversations[active_chat]["title"] = (
             generate_chat_title(prompt))
         
          
         
    with st.chat_message("user"):

      
      
     st.markdown(prompt)

    # Assistant Response

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer,response_time  = get_response(
                question=prompt,
                graph=graph
            )
            print("=" * 80)
            print("ANSWER RETURNED TO STREAMLIT")
            print(answer)
            print("=" * 80)
            print("DEBUG:", response_time)

            st.markdown(answer)
            st.caption(f"⏱️ Response Time: {response_time:.2f} seconds")

    # Save Assistant Message

    messages.append(
        {
            "role": "assistant",
            "content": answer,
            "response_time": response_time
        }
    )
    print("LAST MESSAGE:", messages[-1])