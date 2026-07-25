from backend.database import supabase


def save_message(
    session_id: str,
    role: str,
    message: str,
    response_time: float = None
):
    response = (
        supabase.table("messages")
        .insert({
            "session_id": session_id,
            "role": role,
            "message": message,
            "response_time": response_time
        })
        .execute()
    )

    return response.data[0]



from backend.database import supabase

def get_chat_history(session_id: str):
    response = (
        supabase.table("messages")
        .select("*")
        .eq("session_id", session_id)
        .order("timestamp")
        .execute()
    )

    return response.data