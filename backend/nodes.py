"""
IIT Jodhpur V1 — LangGraph Nodes

Production flow:

    Conversation Resolver
        ↓
    Dense + BM25
        ↓
    Weighted RRF
        ↓
    Deduplication
        ↓
    Local Context Expansion
        ↓
    Conservative Reranking
        ↓
    Evidence Sufficiency
        ↓
    Evidence Coverage
        ↓
    Final Context
        ↓
    One Answer LLM Call
        ↓
    Answer Guard
"""

from backend.state import GraphState

from backend.retriever import (
    dense_retrieve,
    keyword_retrieve,
    reciprocal_rank_fusion,
    deduplicate_documents,
    rerank_documents,
    FINAL_CONTEXT_DOCUMENTS,
    format_context,
)

from backend.local_context import (
    expand_local_context,
)

from backend.evidence import (
    assess_evidence_sufficiency,
)

from backend.evidence_coverage import (
    assess_evidence_coverage,
)

from backend.llm import answer_llm
from backend.prompts import answer_prompt
from backend.answer_guard import guard_answer
from backend.conversation_resolver import resolve_conversation


# =========================================================
# Answer Chain
# =========================================================

answer_chain = answer_prompt | answer_llm


# =========================================================
# Conversation Resolution
# =========================================================

def resolve_conversation_node(
    state: GraphState,
) -> GraphState:
    """
    Resolve a follow-up question only when recent conversation
    context is required.
    """

    result = resolve_conversation(
        question=state["question"],
        chat_history=state.get(
            "chat_history",
            [],
        ),
    )

    return {
        "resolved_question": (
            result["resolved_question"]
        ),
        "conversation_mode": (
            result["mode"]
        ),
        "active_topic": (
            result["active_topic"]
        ),
        "active_entity": (
            result["active_entity"]
        ),
    }


# =========================================================
# Hybrid Retrieval
# =========================================================

def hybrid_retrieve(
    state: GraphState,
) -> GraphState:
    """
    Retrieve evidence using Dense + BM25.
    """

    question = state.get(
        "resolved_question",
        state["question"],
    )

    dense_docs = dense_retrieve(
        question
    )

    keyword_docs = keyword_retrieve(
        question
    )

    return {
        "retrieval_results": [
            dense_docs,
            keyword_docs,
        ],
    }


# =========================================================
# Fuse Documents
# =========================================================

def fuse_retrieved_documents(
    state: GraphState,
) -> GraphState:
    """
    Fuse Dense and BM25 candidates using weighted RRF,
    then remove duplicates.
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
# Local Context Expansion
# =========================================================

def expand_retrieved_context(
    state: GraphState,
) -> GraphState:
    """
    Recover small amounts of useful neighboring context for the
    fused evidence.

    This stage is deterministic and does not add an LLM call.

    The original retrieved chunks remain the anchors. Expansion is
    source-aware, bounded, and rejects obvious noise or structural
    boundaries.
    """

    expanded_docs = expand_local_context(
        state.get(
            "fused_docs",
            [],
        )
    )

    return {
        "expanded_docs": expanded_docs,
    }


# =========================================================
# Reranking
# =========================================================

def rerank_retrieved_documents(
    state: GraphState,
) -> GraphState:
    """
    Rerank the expanded evidence so newly recovered context must
    compete on actual query relevance.

    This prevents local expansion from automatically becoming final
    evidence merely because it is adjacent to a retrieved chunk.
    """

    question = state.get(
        "resolved_question",
        state["question"],
    )

    reranked_docs = rerank_documents(
        query=question,
        documents=state.get(
            "expanded_docs",
            state.get(
                "fused_docs",
                [],
            ),
        ),
        top_k=FINAL_CONTEXT_DOCUMENTS,
    )

    return {
        "reranked_docs":
            reranked_docs,
    }


# =========================================================
# Evidence Sufficiency
# =========================================================

def assess_evidence_node(
    state: GraphState,
) -> GraphState:
    """
    Determine whether the reranked evidence is sufficient to answer
    the question.
    """

    question = state.get(
        "resolved_question",
        state["question"],
    )

    documents = state.get(
        "reranked_docs",
        [],
    )

    result = assess_evidence_sufficiency(
        query=question,
        documents=documents,
    )

    return {
        "evidence_status": (
            result["status"]
        ),
        "evidence_score": (
            result["score"]
        ),
        "relevant_evidence_documents": (
            result["relevant_documents"]
        ),
    }


# =========================================================
# Evidence Coverage
# =========================================================

def assess_evidence_coverage_node(
    state: GraphState,
) -> GraphState:
    """
    Determine whether the evidence is broad enough for the question
    type.
    """

    question = state.get(
        "resolved_question",
        state["question"],
    )

    documents = state.get(
        "reranked_docs",
        [],
    )

    result = assess_evidence_coverage(
        query=question,
        documents=documents,
    )

    return {
        "evidence_coverage_status": (
            result["status"]
        ),
        "evidence_question_type": (
            result["question_type"]
        ),
        "evidence_strong_documents": (
            result["strong_documents"]
        ),
        "evidence_partial_documents": (
            result["partial_documents"]
        ),
        "evidence_combined_characters": (
            result["combined_characters"]
        ),
    }


# =========================================================
# Final Context
# =========================================================

def compress_context(
    state: GraphState,
) -> GraphState:
    """
    Prepare the final evidence context.
    """

    if (
        state.get(
            "evidence_status"
        )
        == "insufficient"
    ):
        return {
            "compressed_docs": []
        }

    compressed_docs = (
        state.get(
            "reranked_docs",
            [],
        )
        [
            :FINAL_CONTEXT_DOCUMENTS
        ]
    )

    return {
        "compressed_docs":
            compressed_docs,
    }


# =========================================================
# Generate Answer
# =========================================================

def generate_answer(
    state: GraphState,
) -> GraphState:
    """
    Generate the final answer from grounded evidence.

    Only the final answer-generation LLM is called here.
    """

    evidence_status = state.get(
        "evidence_status",
        "insufficient",
    )

    coverage_status = state.get(
        "evidence_coverage_status",
        "insufficient",
    )

    # -----------------------------------------------------
    # No usable evidence
    # -----------------------------------------------------

    if (
        evidence_status == "insufficient"
        or coverage_status == "insufficient"
    ):
        return {
            "answer": (
                "I'm sorry, I don't know "
                "based on the available information."
            ),
            "context": "",
            "answer_guard_status": "safe",
            "answer_guard_reason": (
                "insufficient_evidence"
            ),
        }

    # -----------------------------------------------------
    # Prepare context
    # -----------------------------------------------------

    context = format_context(
        state.get(
            "compressed_docs",
            [],
        )
    )

    question_type = state.get(
        "evidence_question_type",
        "descriptive",
    )

    # -----------------------------------------------------
    # One final answer LLM call
    # -----------------------------------------------------

    response = answer_chain.invoke(
        {
            "context": context,
            "question": state.get(
                "resolved_question",
                state["question"],
            ),
            "chat_history": state.get(
                "chat_history",
                [],
            ),
            "question_type": question_type,
            "evidence_coverage": coverage_status,
        }
    )

    # -----------------------------------------------------
    # Answer Guard
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