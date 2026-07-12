from backend.state import GraphState
from backend.llm import llm

from backend.prompts import (
    rewrite_prompt,
    multi_query_prompt,
    answer_prompt,
)

from backend.retriever import (
    dense_retrieve,
    keyword_retrieve,
    reciprocal_rank_fusion,
    FINAL_CONTEXT_DOCUMENTS,
    format_context,
)

# =========================================================
# Chains
# =========================================================

rewrite_chain = rewrite_prompt | llm
multi_query_chain = multi_query_prompt | llm
answer_chain = answer_prompt | llm


# =========================================================
# Rewrite Query
# =========================================================

def rewrite_query(state: GraphState) -> GraphState:

    rewritten_question = rewrite_chain.invoke(
        {
            "question": state["question"],
            "chat_history": state["chat_history"]
        }
    ).content.strip()

    print("\n" + "=" * 80)
    print("REWRITE QUERY")
    print("=" * 80)
    print("Original Question :")
    print(state["question"])
    print()
    print("Rewritten Question :")
    print(rewritten_question)
    print("=" * 80)

    return {
        "rewritten_question": rewritten_question
    }


# =========================================================
# Parse Generated Queries
# =========================================================

def parse_generated_queries(response: str) -> list[str]:

    queries = []

    for line in response.split("\n"):

        line = line.strip()

        if line:
            queries.append(line)

    return queries


# =========================================================
# Generate Multi Query
# =========================================================

def generate_multi_query(state: GraphState) -> GraphState:

    response = multi_query_chain.invoke(
        {
            "rewritten_question": state["rewritten_question"]
        }
    )

    generated_queries = parse_generated_queries(
        response.content
    )

    print("\n" + "=" * 80)
    print("GENERATED SEARCH QUERIES")
    print("=" * 80)

    for index, query in enumerate(generated_queries, start=1):
        print(f"{index}. {query}")

    print("=" * 80)

    return {
        "generated_queries": generated_queries
    }


# =========================================================
# Hybrid Retrieval
# =========================================================

def hybrid_retrieve(state: GraphState) -> GraphState:

    retrieval_results = []

    print("\n" + "=" * 80)
    print("HYBRID RETRIEVAL")
    print("=" * 80)

    for index, query in enumerate(state["generated_queries"], start=1):

        print(f"\nQuery {index}: {query}")

        dense_docs = dense_retrieve(query)
        keyword_docs = keyword_retrieve(query)

        retrieval_results.append(dense_docs)
        retrieval_results.append(keyword_docs)

        print(f"Dense Retrieved   : {len(dense_docs)} documents")
        print(f"Keyword Retrieved : {len(keyword_docs)} documents")

    print("=" * 80)

    return {
        "retrieval_results": retrieval_results
    }


# =========================================================
# Fuse Documents
# =========================================================

def fuse_retrieved_documents(state: GraphState) -> GraphState:

    fused_docs = reciprocal_rank_fusion(
        state["retrieval_results"]
    )

    print("\n" + "=" * 80)
    print("RRF FUSION")
    print("=" * 80)
    print(f"Documents after Fusion : {len(fused_docs)}")
    print("=" * 80)

    return {
        "fused_docs": fused_docs
    }


# =========================================================
# Compress Context
# =========================================================

def compress_context(state: GraphState) -> GraphState:

    compressed_docs = state["fused_docs"][:FINAL_CONTEXT_DOCUMENTS]

    print("\n" + "=" * 80)
    print("CONTEXT COMPRESSION")
    print("=" * 80)
    print(f"Keeping Top {len(compressed_docs)} Documents")
    print("=" * 80)

    return {
        "compressed_docs": compressed_docs
    }


# =========================================================
# Generate Answer
# =========================================================

def generate_answer(state: GraphState) -> GraphState:

    context = format_context(
        state["compressed_docs"]
    )

    print("\n" + "=" * 80)
    print("FINAL CONTEXT SENT TO LLM")
    print("=" * 80)
    print(context[:1500])   # Prevent terminal flooding
    print("=" * 80)

    response = answer_chain.invoke(
        {
            "context": context,
            "question": state["question"],
            "chat_history": state["chat_history"]
        }
    )

    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)
    print(response.content.strip())
    print("=" * 80)

    return {
        "answer": response.content.strip()
    }