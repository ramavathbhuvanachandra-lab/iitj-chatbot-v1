"""
IIT Jodhpur V1 — LangGraph Nodes

Purpose
-------
Define the retrieval and answer-generation nodes used by the
production chatbot graph.

Current pipeline:

    User Question
        ↓
    Dense + BM25 Retrieval
        ↓
    Weighted RRF
        ↓
    Deduplication
        ↓
    Top-5 Evidence
        ↓
    Answer Generation
        ↓
    Deterministic Answer Guard
        ↓
    Final Answer

Important invariants
--------------------
- The RAG retrieval architecture is unchanged.
- Retrieved evidence remains the factual source for answers.
- Conversation history is passed to the answer model only for
  understanding references.
- The final model output always passes through Answer Guard.
"""

from backend.state import GraphState

from backend.retriever import (
    dense_retrieve,
    keyword_retrieve,
    reciprocal_rank_fusion,
    deduplicate_documents,
    FINAL_CONTEXT_DOCUMENTS,
    format_context,
)

from backend.llm import llm
from backend.prompts import answer_prompt
from backend.answer_guard import guard_answer


# =========================================================
# Answer Chain
# =========================================================

answer_chain = answer_prompt | llm


# =========================================================
# Hybrid Retrieval
# =========================================================

def hybrid_retrieve(
    state: GraphState,
) -> GraphState:
    """
    Retrieve evidence directly from the user's question.

    Production retrieval path:

        Original question
            ↓
        Dense + BM25
            ↓
        Weighted RRF
    """

    question = state["question"]

    dense_docs = dense_retrieve(
        question
    )

    keyword_docs = keyword_retrieve(
        question
    )

    retrieval_results = [
        dense_docs,
        keyword_docs,
    ]

    return {
        "retrieval_results": retrieval_results,
    }


# =========================================================
# Fuse Documents
# =========================================================

def fuse_retrieved_documents(
    state: GraphState,
) -> GraphState:
    """
    Fuse Dense and BM25 candidates using weighted RRF.
    """

    fused_docs = reciprocal_rank_fusion(
        state["retrieval_results"]
    )

    fused_docs = deduplicate_documents(
        fused_docs
    )

    return {
        "fused_docs": fused_docs,
    }


# =========================================================
# Final Evidence
# =========================================================

def compress_context(
    state: GraphState,
) -> GraphState:
    """
    Select the final evidence chunks.

    Top 5 remains unchanged from the current V1 retrieval design.
    """

    compressed_docs = (
        state["fused_docs"]
        [:FINAL_CONTEXT_DOCUMENTS]
    )

    return {
        "compressed_docs": compressed_docs,
    }


# =========================================================
# Generate Answer
# =========================================================

def generate_answer(
    state: GraphState,
) -> GraphState:
    """
    Generate an answer from the final retrieved evidence.

    The raw model output is passed through the deterministic
    Answer Guard before being returned to GraphState.
    """

    context = format_context(
        state["compressed_docs"]
    )

    response = answer_chain.invoke(
        {
            "context": context,
            "question": state["question"],
            "chat_history": state.get(
                "chat_history",
                [],
            ),
        }
    )

    # -----------------------------------------------------
    # Deterministic output-contract enforcement
    # -----------------------------------------------------

    guard_result = guard_answer(
        response.content
    )

    return {
        "answer": guard_result["answer"],
        "context": context,
        "answer_guard_status": (
            guard_result["status"]
        ),
        "answer_guard_reason": (
            guard_result["reason"]
        ),
    }