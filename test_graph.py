from backend.graph import create_graph

graph = create_graph()


question = "What research areas are available in the Electrical Engineering department"

state = {
    "question": question
}

result = graph.invoke(state)

print(result["answer"])