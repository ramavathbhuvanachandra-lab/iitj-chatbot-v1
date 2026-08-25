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

from backend.evidence import (
    assess_evidence_sufficiency,
)

from backend.evidence_coverage import (
    assess_evidence_coverage,
)

from backend.llm import llm
from backend.prompts import answer_prompt
from backend.answer_guard import guard_answer
from backend.conversation_resolver import resolve_conversation


# =========================================================
# Answer Chain
# =========================================================

answer_chain = answer_prompt | llm


# =========================================================
# Conversation Resolution
# =========================================================

def resolve_conversation_node(
    state: GraphState,
) -> GraphState:

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
# Reranking
# =========================================================

def rerank_retrieved_documents(
    state: GraphState,
) -> GraphState:

    question = state.get(
        "resolved_question",
        state["question"],
    )

    reranked_docs = rerank_documents(
        query=question,
        documents=state["fused_docs"],
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
    # IMPORTANT:
    # No extra LLM call is created here.
    #
    # These are deterministic signals generated earlier in
    # the graph and passed into the SAME answer-generation call.
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