from backend.assistant_router import assistant_router

questions = [

    

    "Where is the Hostel Office?",
    "Hostel fees",
]

for q in questions:
    print("=" * 80)
    print("Question:", q)
    print(assistant_router(q))