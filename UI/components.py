import streamlit as st


# ---------------------------------------------------------
# Section Divider
# ---------------------------------------------------------

def divider():

    st.divider()


# ---------------------------------------------------------
# Section Title
# ---------------------------------------------------------

def section_title(title: str):

    st.subheader(title)


# ---------------------------------------------------------
# Information Box
# ---------------------------------------------------------

def info_box(message: str):

    st.info(message)


# ---------------------------------------------------------
# Success Box
# ---------------------------------------------------------

def success_box(message: str):

    st.success(message)


# ---------------------------------------------------------
# Warning Box
# ---------------------------------------------------------

def warning_box(message: str):

    st.warning(message)


# ---------------------------------------------------------
# Error Box
# ---------------------------------------------------------

def error_box(message: str):

    st.error(message)

# ---------------------------------------------------------
# Quick Action Cards
# ---------------------------------------------------------

QUICK_ACTIONS = {
    "🎓 Admission":
        "Provide complete information about the IIT Jodhpur admission process, required documents, registration, and joining procedure.",

    "🏠 Hostel":
        "Provide complete information about IIT Jodhpur hostels, hostel allocation, facilities, mess, Wi-Fi, laundry, and hostel rules.",

    "💰 Fees":
        "Provide complete information about the IIT Jodhpur fee structure, fee payment process, scholarships, and refund policy.",

    "🏛 Departments":
        "Provide an overview of all academic departments and schools at IIT Jodhpur along with their major research areas."
}


def quick_action_cards():
    """
    Display quick action buttons.

    Returns:
        str | None
            The predefined question corresponding to the clicked button.
    """

    st.markdown("### Quick Actions")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎓 Admission", use_container_width=True):
            return QUICK_ACTIONS["🎓 Admission"]

        if st.button("💰 Fees", use_container_width=True):
            return QUICK_ACTIONS["💰 Fees"]

    with col2:
        if st.button("🏠 Hostel", use_container_width=True):
            return QUICK_ACTIONS["🏠 Hostel"]

        if st.button("🏛 Departments", use_container_width=True):
            return QUICK_ACTIONS["🏛 Departments"]

    return None


