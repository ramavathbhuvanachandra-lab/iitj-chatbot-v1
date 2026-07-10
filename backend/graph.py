from langgraph.graph import StateGraph, START, END

from backend.state import GraphState
from backend.nodes import (
    rewrite_query,
    generate_multi_query,
    hybrid_retrieve,
    fuse_retrieved_documents,
    compress_context,
    generate_answer,
)


def create_graph():
    """
    Create and compile the LangGraph workflow.
    """

    workflow = StateGraph(GraphState)

    # -------------------------
    # Register Nodes
    # -------------------------

    workflow.add_node(
        "rewrite_query",
        rewrite_query,
    )

    workflow.add_node(
        "generate_multi_query",
        generate_multi_query,
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
        "compress_context",
        compress_context,
    )

    workflow.add_node(
        "generate_answer",
        generate_answer,
    )

    # -------------------------
    # Connect Nodes
    # -------------------------

    workflow.add_edge(
        START,
        "rewrite_query",
    )

    workflow.add_edge(
        "rewrite_query",
        "generate_multi_query",
    )

    workflow.add_edge(
        "generate_multi_query",
        "hybrid_retrieve",
    )

    workflow.add_edge(
        "hybrid_retrieve",
        "fuse_retrieved_documents",
    )

    workflow.add_edge(
        "fuse_retrieved_documents",
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