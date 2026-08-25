"""
IIT Jodhpur V1 — Evidence Coverage

Purpose
-------
Estimate whether retrieved evidence is likely to cover the scope
of the user's question.

This layer is deterministic.

It does NOT:
- call an LLM
- perform retrieval
- rewrite the question
- enforce hard document filters

Classification:

    supported
    partially_supported
    insufficient
"""

import re
from typing import Any, Dict

from backend.retriever import (
    normalize_text,
    detect_programs,
    detect_topics,
    detect_entities,
)


# =========================================================
# Configuration
# =========================================================

STRONG_DOCUMENT_SCORE = 0.55
PARTIAL_DOCUMENT_SCORE = 0.30

MIN_LIST_CONTENT_CHARS = 450


# =========================================================
# Question Type Detection
# =========================================================

def detect_question_type(
    question: str,
) -> str:
    """
    Detect the broad type of user question.

    Priority:

        quantitative
        requirements
        list
        descriptive
    """

    normalized = normalize_text(
        question
    )

    # -----------------------------------------------------
    # Quantitative / fee questions
    # -----------------------------------------------------

    quantitative_patterns = [
        r"\bhow many\b",
        r"\bhow much\b",
        r"\bwhat is the fee\b",
        r"\bwhat are the fees\b",
        r"\bwhat are .* fees\b",
        r"\bwhat is the cost\b",
        r"\bwhat are .* costs\b",
        r"\bhow long\b",
        r"\bwhat is the amount\b",
        r"\bwhat is the charge\b",
        r"\bwhat are the charges\b",
    ]

    if any(
        re.search(
            pattern,
            normalized,
        )
        for pattern in quantitative_patterns
    ):
        return "quantitative"

    # -----------------------------------------------------
    # Requirement / eligibility questions
    # -----------------------------------------------------

    requirement_patterns = [
        r"\beligibility\b",
        r"\beligible\b",
        r"\brequirements\b",
        r"\bqualification\b",
        r"\bcriteria\b",
        r"\bwhat do i need\b",
        r"\bwhat is required\b",
        r"\bwhat are .* requirements\b",
    ]

    if any(
        re.search(
            pattern,
            normalized,
        )
        for pattern in requirement_patterns
    ):
        return "requirements"

    # -----------------------------------------------------
    # List / enumeration questions
    # -----------------------------------------------------

    list_patterns = [
        r"\bwhat research areas\b",
        r"\bwhat areas\b",
        r"\bwhat facilities\b",
        r"\bwhat programs\b",
        r"\bwhat courses\b",
        r"\bwhat departments\b",
        r"\bwhat are the .* available\b",
        r"\bwhich .* are available\b",
        r"\bwhat are the main\b",
        r"\blist\b",
    ]

    if any(
        re.search(
            pattern,
            normalized,
        )
        for pattern in list_patterns
    ):
        return "list"

    return "descriptive"


# =========================================================
# Query Anchors
# =========================================================

STOPWORDS = {
    "what",
    "which",
    "where",
    "when",
    "how",
    "who",
    "why",
    "are",
    "is",
    "the",
    "a",
    "an",
    "of",
    "for",
    "in",
    "on",
    "to",
    "at",
    "do",
    "does",
    "can",
    "could",
    "would",
    "should",
    "there",
    "available",
    "please",
    "tell",
    "me",
}


def _query_anchors(
    question: str,
):
    """
    Extract lightweight lexical anchors from the question.
    """

    normalized = normalize_text(
        question
    )

    return {
        token
        for token in normalized.split()
        if (
            len(token) > 2
            and token not in STOPWORDS
        )
    }


# =========================================================
# Requirement Evidence
# =========================================================

REQUIREMENT_CONTENT_TERMS = {
    "must",
    "minimum",
    "marks",
    "cgpa",
    "degree",
    "bachelor",
    "bachelors",
    "master",
    "masters",
    "qualification",
    "qualifications",
    "eligible",
    "eligibility",
    "requirement",
    "requirements",
    "criteria",
    "gate",
    "experience",
    "admitted",
    "admission",
}


def _requirement_evidence_signal(
    content: str,
) -> float:
    """
    Detect substantive eligibility/requirement language.

    This uses structural requirement language rather than
    requiring exact wording from the question.
    """

    normalized = normalize_text(
        content
    )

    matched = {
        term
        for term in REQUIREMENT_CONTENT_TERMS
        if term in normalized
    }

    if not matched:
        return 0.0

    return min(
        len(matched) * 0.07,
        0.35,
    )


# =========================================================
# Source-aware signals
# =========================================================

def _source_text(
    document,
) -> str:
    """
    Normalize source metadata for semantic scope detection.

    This is a supporting signal only.
    """

    source = document.metadata.get(
        "source",
        "",
    )

    return normalize_text(
        source
    )


# =========================================================
# Document Support Summary
# =========================================================

def _document_support(
    question_text: str,
    document,
) -> Dict[str, Any]:
    """
    Summarize how strongly one document supports the question.

    Both content and source metadata participate in scope
    detection.
    """

    query = normalize_text(
        question_text
    )

    content = normalize_text(
        document.page_content
    )

    source = _source_text(
        document
    )

    combined_evidence_text = (
        f"{source} {content}"
    ).strip()

    if not content:
        return {
            "score": 0.0,
            "characters": 0,
            "matched_programs": set(),
            "matched_topics": set(),
            "matched_entities": set(),
            "requirement_signal": 0.0,
        }

    query_tokens = _query_anchors(
        query
    )

    content_tokens = {
        token
        for token in content.split()
        if len(token) > 2
    }

    score = 0.0

    # -----------------------------------------------------
    # 1. Lexical overlap
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
    # 2. Program alignment
    # -----------------------------------------------------

    query_programs = detect_programs(
        query
    )

    document_programs = detect_programs(
        combined_evidence_text
    )

    matched_programs = (
        query_programs
        & document_programs
    )

    if matched_programs:
        score += 0.20

    # -----------------------------------------------------
    # 3. Topic alignment
    # -----------------------------------------------------

    query_topics = detect_topics(
        query
    )

    document_topics = detect_topics(
        combined_evidence_text
    )

    matched_topics = (
        query_topics
        & document_topics
    )

    if matched_topics:
        score += 0.15

    # -----------------------------------------------------
    # 4. Entity alignment
    # -----------------------------------------------------

    query_entities = detect_entities(
        query
    )

    document_entities = detect_entities(
        combined_evidence_text
    )

    matched_entities = (
        query_entities
        & document_entities
    )

    if matched_entities:
        score += 0.15

    # -----------------------------------------------------
    # 5. Requirement evidence
    # -----------------------------------------------------

    requirement_signal = (
        _requirement_evidence_signal(
            combined_evidence_text
        )
    )

    score += requirement_signal

    return {
        "score": min(
            score,
            1.0,
        ),
        "characters": len(
            content
        ),
        "matched_programs": (
            matched_programs
        ),
        "matched_topics": (
            matched_topics
        ),
        "matched_entities": (
            matched_entities
        ),
        "requirement_signal": (
            requirement_signal
        ),
    }


# =========================================================
# Coverage Assessment
# =========================================================

def assess_evidence_coverage(
    query: str,
    documents,
) -> Dict[str, Any]:
    """
    Estimate whether the evidence is likely to cover the scope
    of the user's question.
    """

    question_type = detect_question_type(
        query
    )

    # -----------------------------------------------------
    # Empty evidence
    # -----------------------------------------------------

    if not documents:

        return {
            "status": "insufficient",
            "question_type": question_type,
            "strong_documents": 0,
            "partial_documents": 0,
            "combined_characters": 0,
        }

    # -----------------------------------------------------
    # Score documents
    # -----------------------------------------------------

    summaries = [
        _document_support(
            question_text=query,
            document=document,
        )
        for document in documents
    ]

    strong_documents = sum(
        1
        for item in summaries
        if item["score"]
        >= STRONG_DOCUMENT_SCORE
    )

    partial_documents = sum(
        1
        for item in summaries
        if item["score"]
        >= PARTIAL_DOCUMENT_SCORE
    )

    combined_characters = sum(
        item["characters"]
        for item in summaries
    )

    # -----------------------------------------------------
    # No meaningful evidence
    # -----------------------------------------------------

    if partial_documents == 0:

        return {
            "status": "insufficient",
            "question_type": question_type,
            "strong_documents": 0,
            "partial_documents": 0,
            "combined_characters": (
                combined_characters
            ),
        }

    # -----------------------------------------------------
    # Descriptive
    # -----------------------------------------------------

    if question_type == "descriptive":

        status = (
            "supported"
            if strong_documents >= 1
            else "partially_supported"
        )

        return {
            "status": status,
            "question_type": question_type,
            "strong_documents": strong_documents,
            "partial_documents": partial_documents,
            "combined_characters": (
                combined_characters
            ),
        }

    # -----------------------------------------------------
    # Quantitative
    # -----------------------------------------------------

    if question_type == "quantitative":

        if (
            strong_documents >= 1
            and combined_characters >= 250
        ):
            status = "supported"

        elif partial_documents >= 1:
            status = "partially_supported"

        else:
            status = "insufficient"

        return {
            "status": status,
            "question_type": question_type,
            "strong_documents": strong_documents,
            "partial_documents": partial_documents,
            "combined_characters": (
                combined_characters
            ),
        }

    # -----------------------------------------------------
    # Requirements
    # -----------------------------------------------------

    if question_type == "requirements":

        requirement_strong_documents = sum(
            1
            for item in summaries
            if (
                item["requirement_signal"]
                >= 0.20
            )
        )

        # -------------------------------------------------
        # Program-specific requirement evidence is strong
        # enough on its own.
        #
        # We deliberately do NOT impose a character-count
        # threshold here. A short but highly specific chunk
        # can legitimately contain the complete requirement
        # needed for a question.
        # -------------------------------------------------

        program_aligned_requirement = sum(
            1
            for item in summaries
            if (
                item["requirement_signal"]
                >= 0.20
                and bool(
                    item["matched_programs"]
                )
            )
        )

        if (
            strong_documents >= 1
            or program_aligned_requirement >= 1
            or (
                requirement_strong_documents >= 1
                and partial_documents >= 1
            )
        ):
            status = "supported"

        elif partial_documents >= 1:
            status = "partially_supported"

        else:
            status = "insufficient"

        return {
            "status": status,
            "question_type": question_type,
            "strong_documents": strong_documents,
            "partial_documents": partial_documents,
            "combined_characters": (
                combined_characters
            ),
        }

    # -----------------------------------------------------
    # List
    # -----------------------------------------------------

    if question_type == "list":

        if (
            strong_documents >= 1
            and combined_characters
            >= MIN_LIST_CONTENT_CHARS
        ):
            status = "supported"

        elif partial_documents >= 1:
            status = "partially_supported"

        else:
            status = "insufficient"

        return {
            "status": status,
            "question_type": question_type,
            "strong_documents": strong_documents,
            "partial_documents": partial_documents,
            "combined_characters": (
                combined_characters
            ),
        }

    # -----------------------------------------------------
    # Fallback
    # -----------------------------------------------------

    return {
        "status": "partially_supported",
        "question_type": question_type,
        "strong_documents": strong_documents,
        "partial_documents": partial_documents,
        "combined_characters": (
            combined_characters
        ),
    }