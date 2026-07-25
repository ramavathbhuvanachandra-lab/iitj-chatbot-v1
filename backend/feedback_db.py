from backend.database import supabase


def save_feedback(
    message_id: str,
    session_id: str,
    feedback: str,
):
    response = (
        supabase.table("message_feedback")
        .insert(
            {
                "message_id": message_id,
                "session_id": session_id,
                "feedback": feedback,
            }
        )
        .execute()
    )

    return response.data[0]