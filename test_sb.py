from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen2.5:3b",
    base_url="http://localhost:11434"
)

print(llm.invoke("Hello"))