from typing import TypedDict, List, Dict, Any

from langchain_core.documents import Document


class GraphState(TypedDict):
    # --------------------------------------------------------
    # User Input
    # --------------------------------------------------------
    question: str
    chat_history: List[Dict[str, Any]]

    # --------------------------------------------------------
    # Query Processing
    # --------------------------------------------------------
    rewritten_question: str
    generated_queries: List[str]

    # --------------------------------------------------------
    # Retrieval Pipeline
    # --------------------------------------------------------
    retrieval_results: List[List[Document]]
    fused_docs: List[Document]
    reranked_docs: List[Document]
    compressed_docs: List[Document]

    # --------------------------------------------------------
    # Final Output
    # --------------------------------------------------------
    answer: str

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------
    hallucination_score: float
    answer_score: float