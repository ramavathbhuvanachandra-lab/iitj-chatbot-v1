from typing import TypedDict, List, Dict, Any

from langchain_core.documents import Document


class GraphState(TypedDict, total=False):

    # =========================================================
    # User Input
    # =========================================================

    question: str
    chat_history: List[Dict[str, Any]]

    # =========================================================
    # Conversation Resolution
    # =========================================================

    resolved_question: str
    conversation_mode: str
    active_topic: str
    active_entity: str

    # =========================================================
    # Retrieval
    # =========================================================

    retrieval_results: List[List[Document]]
    retrieval_weights: List[float]

    fused_docs: List[Document]
    reranked_docs: List[Document]
    compressed_docs: List[Document]

    # =========================================================
    # Evidence
    # =========================================================

    evidence_status: str
    evidence_score: float
    relevant_evidence_documents: int

    evidence_coverage_status: str
    evidence_question_type: str
    evidence_strong_documents: int
    evidence_partial_documents: int
    evidence_combined_characters: int

    # =========================================================
    # Final Answer
    # =========================================================

    answer: str
    context: str

    # =========================================================
    # Evaluation
    # =========================================================

    hallucination_score: float
    answer_score: float

    # =========================================================
    # Answer Guard
    # =========================================================

    answer_guard_status: str
    answer_guard_reason: str