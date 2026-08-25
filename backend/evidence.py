"""
IIT Jodhpur V1 — Evidence Sufficiency

Purpose
-------
Determine whether the retrieved evidence is sufficiently aligned
with the user's question before answer generation.

Design principles
-----------------
- Deterministic and lightweight.
- No additional LLM call.
- Uses existing program/topic/entity signals.
- Soft scoring, not brittle hard filtering.
- Conservative: weak or empty evidence is marked insufficient.
"""

from typing import Dict, Any

from backend.retriever import (
    normalize_text,
    detect_programs,
    detect_topics,
    detect_entities,
)


# =========================================================
# Configuration
# =========================================================

MIN_SUPPORTED_SCORE = 0.30

MIN_RELEVANT_DOCUMENTS = 1


# =========================================================
# Single-document support
# =========================================================

def _score_document_support(
    query: str,
    document,
) -> float:
    """
    Estimate how directly one document supports the query.
    """

    normalized_query = normalize_text(
        query
    )

    content = normalize_text(
        document.page_content
    )

    # -----------------------------------------------------
    # Empty content
    # -----------------------------------------------------

    if not content:
        return 0.0

    query_tokens = {
        token
        for token in normalized_query.split()
        if len(token) > 2
    }

    content_tokens = {
        token
        for token in content.split()
        if len(token) > 2
    }

    score = 0.0

    # -----------------------------------------------------
    # Query overlap
    # -----------------------------------------------------

    if query_tokens:

        overlap = (
            len(
                query_tokens
                & content_tokens
            )
            / len(query_tokens)
        )

        score += (
            overlap
            * 0.50
        )

    # -----------------------------------------------------
    # Program alignment
    # -----------------------------------------------------

    query_programs = detect_programs(
        query
    )

    document_programs = detect_programs(
        content
    )

    if query_programs:

        matched_programs = (
            query_programs
            & document_programs
        )

        if matched_programs:
            score += 0.25

    # -----------------------------------------------------
    # Topic alignment
    # -----------------------------------------------------

    query_topics = detect_topics(
        query
    )

    document_topics = detect_topics(
        content
    )

    if query_topics:

        matched_topics = (
            query_topics
            & document_topics
        )

        if matched_topics:
            score += 0.15

    # -----------------------------------------------------
    # Entity alignment
    # -----------------------------------------------------

    query_entities = detect_entities(
        query
    )

    document_entities = detect_entities(
        content
    )

    if query_entities:

        matched_entities = (
            query_entities
            & document_entities
        )

        if matched_entities:
            score += 0.10

    return min(
        score,
        1.0,
    )


# =========================================================
# Evidence assessment
# =========================================================

def assess_evidence_sufficiency(
    query: str,
    documents,
) -> Dict[str, Any]:
    """
    Assess whether retrieved evidence is sufficient to answer
    the requested question.

    Returns:

        {
            "status": "supported" | "insufficient",
            "score": float,
            "relevant_documents": int,
        }
    """

    if not documents:
        return {
            "status": "insufficient",
            "score": 0.0,
            "relevant_documents": 0,
        }

    document_scores = []

    for document in documents:

        score = _score_document_support(
            query=query,
            document=document,
        )

        document_scores.append(
            score
        )

    relevant_documents = sum(
        1
        for score in document_scores
        if score >= MIN_SUPPORTED_SCORE
    )

    best_score = max(
        document_scores,
        default=0.0,
    )

    if (
        relevant_documents
        >= MIN_RELEVANT_DOCUMENTS
        and best_score
        >= MIN_SUPPORTED_SCORE
    ):
        status = "supported"

    else:
        status = "insufficient"

    return {
        "status": status,
        "score": round(
            best_score,
            3,
        ),
        "relevant_documents": (
            relevant_documents
        ),
    }