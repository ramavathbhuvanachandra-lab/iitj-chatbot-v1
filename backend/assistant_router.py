from backend.campus_navigation import find_location
from backend.emergency import find_emergency


def assistant_router(question: str):
    """
    Route the question to structured modules first.
    """

    structured = []

    location = find_location(question)
    if location:
        structured.append({
            "type": "navigation",
            "data": location
        })

    emergency = find_emergency(question)
    if emergency:
        structured.append({
            "type": "emergency",
            "data": emergency
        })

    # Simple V1 rule
    question_lower = question.lower()

    question_lower = question.lower()

    information_keywords = [
    "timing",
    "timings",
    "time",
    "hours",
    "working hours",
    "open",
    "close",
    "fee",
    "fees",
    "admission",
    "hostel",
    "facility",
    "facilities",
    "department",
    "course",
    "syllabus",
    "process",
    "procedure",
    "rules",
    "eligibility",
    "contact",
    "email",
    "what",
    "when",
    ]

    need_rag = any(
        keyword in question_lower
        for keyword in information_keywords
    )

    # If nothing was found in structured data,
   # always use RAG.
    if not structured:
       need_rag = True

    return {
        "structured": structured,
        "need_rag": need_rag,
    }

