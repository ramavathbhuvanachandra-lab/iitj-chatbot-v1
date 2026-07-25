import uuid
from backend.database import supabase

def create_session(user_id: str):
    session_id = f"SES_{uuid.uuid4().hex[:8].upper()}"

    response = (
        supabase.table("sessions")
        .insert({
            "session_id": session_id,
            "user_id": user_id
        })
        .execute()
    )

    return response.data[0]