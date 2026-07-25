from backend.assistant_router import assistant_router

questions = [

    # Navigation
    "Where is the library?",
    "How do I reach the library?",
    "Can you guide me to the Central Mess?",
    "Where is the canteen?",
    "Show me the food court.",

    # Emergency
    "Ambulance number",
    "I need a doctor urgently.",
    "Where is the Medical Centre?",
    "Security office phone number",
    "I need security help.",

    # Pure RAG
    "Admission process",
    "Hostel fees",
    "Electrical Engineering syllabus",
    "Library timings",
    "Mess timings",

    # Mixed Questions
    "Where is the library and what are its timings?",
    "Where is the Central Mess and what are the mess timings?",
    "Where is the Medical Centre and what are its working hours?",
    "Give me the Security Office number and explain when I should contact them.",
    "How do I reach the library and what facilities are available there?",
    "Where is the hostel office and how can I apply for hostel allocation?",
]

for q in questions:
    print("=" * 80)
    print("Question:", q)
    print(assistant_router(q))