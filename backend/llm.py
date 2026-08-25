import os

from langchain_ollama import ChatOllama


# =========================================================
# Ollama Configuration
# =========================================================

OLLAMA_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)


# =========================================================
# Query Processing LLM
# =========================================================

query_llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
    base_url=OLLAMA_URL,
)


# =========================================================
# Final Answer LLM
# =========================================================

answer_llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
    base_url=OLLAMA_URL,
)


# =========================================================
# Backward Compatibility
# =========================================================

# Existing rewrite/multi-query code uses `llm`.
llm = query_llm