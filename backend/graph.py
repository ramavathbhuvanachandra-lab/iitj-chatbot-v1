"""
IIT Jodhpur V1 — LangGraph Workflow

Production pipeline:

    START
      ↓
    Conversation Resolver
      ↓
    Hybrid Retrieval
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
    Context Preparation
      ↓
    One Answer LLM Call
      ↓
    Answer Guard
      ↓
    END
"""

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from backend.state import GraphState

from backend.nodes import (
    resolve_conversation_node,
    hybrid_retrieve,
    fuse_retrieved_documents,
    rerank_retrieved_documents,
    assess_evidence_node,
    assess_evidence_coverage_node,
    compress_context,
    generate_answer,
)


# =========================================================
# Create Graph
# =========================================================

def create_graph():

    workflow = StateGraph(
        GraphState
    )

    # -----------------------------------------------------
    # Nodes
    # -----------------------------------------------------

    workflow.add_node(
        "resolve_conversation",
        resolve_conversation_node,
    )

    workflow.add_node(
        "hybrid_retrieve",
        hybrid_retrieve,
    )

    workflow.add_node(
        "fuse_retrieved_documents",
        fuse_retrieved_documents,
    )

    workflow.add_node(
        "rerank_retrieved_documents",
        rerank_retrieved_documents,
    )

    workflow.add_node(
        "assess_evidence",
        assess_evidence_node,
    )

    workflow.add_node(
        "assess_evidence_coverage",
        assess_evidence_coverage_node,
    )

    workflow.add_node(
        "compress_context",
        compress_context,
    )

    workflow.add_node(
        "generate_answer",
        generate_answer,
    )

    # -----------------------------------------------------
    # Edges
    # -----------------------------------------------------

    workflow.add_edge(
        START,
        "resolve_conversation",
    )

    workflow.add_edge(
        "resolve_conversation",
        "hybrid_retrieve",
    )

    workflow.add_edge(
        "hybrid_retrieve",
        "fuse_retrieved_documents",
    )

    workflow.add_edge(
        "fuse_retrieved_documents",
        "rerank_retrieved_documents",
    )

    workflow.add_edge(
        "rerank_retrieved_documents",
        "assess_evidence",
    )

    workflow.add_edge(
        "assess_evidence",
        "assess_evidence_coverage",
    )

    workflow.add_edge(
        "assess_evidence_coverage",
        "compress_context",
    )

    workflow.add_edge(
        "compress_context",
        "generate_answer",
    )

    workflow.add_edge(
        "generate_answer",
        END,
    )

    return workflow.compile()