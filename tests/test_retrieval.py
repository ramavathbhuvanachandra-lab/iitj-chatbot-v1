"""
IIT Jodhpur Chatbot V1 - Retrieval Debug Test

Purpose:
    Inspect the internal retrieval pipeline for a single query.

Pipeline inspected:

    Original Question
        ↓
    Rewrite Query
        ↓
    Generate Multi Query
        ↓
    Dense Retrieval + BM25
        ↓
    Reciprocal Rank Fusion
        ↓
    Context Compression
        ↓
    Final Retrieved Context

This script does NOT modify the production chatbot.

Run from the iitj_v1 directory:

    python tests/test_retrieval.py
"""

import sys
import time
from pathlib import Path


# ============================================================
# Project Path Setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Project Imports
# ============================================================

from backend.nodes import (
    rewrite_query,
    generate_multi_query,
    hybrid_retrieve,
    fuse_retrieved_documents,
    compress_context,
)

from backend.retriever import (
    format_context,
    FINAL_CONTEXT_DOCUMENTS,
)

from backend.state import GraphState


# ============================================================
# Configuration
# ============================================================

QUERY = "What are the eligibility requirements for B.Tech admission at IIT Jodhpur?"

CHAT_HISTORY = []

TOP_K_TO_PRINT = 5
CONTENT_PREVIEW_LENGTH = 1000


# ============================================================
# Helper Functions
# ============================================================

def print_document(label: str, rank: int, doc):
    """
    Print metadata and content preview for one retrieved document.
    """

    metadata = getattr(doc, "metadata", {}) or {}
    content = getattr(doc, "page_content", "") or ""

    print("\n" + "=" * 90)
    print(f"{label} #{rank}")
    print("=" * 90)

    print("SOURCE:")
    print(metadata.get("source", "N/A"))

    print("\nMETADATA:")

    for key, value in metadata.items():
        print(f"  {key}: {value}")

    print("\nCONTENT:")
    print(content[:CONTENT_PREVIEW_LENGTH])

    if len(content) > CONTENT_PREVIEW_LENGTH:
        print("\n[Content truncated]")


def print_timing(step_name: str, start_time: float):
    """
    Print execution time for a pipeline step.
    """

    elapsed = time.perf_counter() - start_time

    print(
        f"\n⏱️ {step_name}: {elapsed:.2f} seconds"
    )

    return elapsed


# ============================================================
# Main Debug Pipeline
# ============================================================

def main():

    print("\n" + "#" * 90)
    print("# IITJ V1 RETRIEVAL DEBUG TEST")
    print("#" * 90)

    print("\nORIGINAL QUESTION:")
    print(QUERY)

    # ========================================================
    # Initial Graph State
    # ========================================================

    state: GraphState = {
        "question": QUERY,
        "chat_history": CHAT_HISTORY,
    }

    # ========================================================
    # STEP 1 — Rewrite Query
    # ========================================================

    print("\n\n" + "#" * 90)
    print("# STEP 1 — QUERY REWRITING")
    print("#" * 90)

    start = time.perf_counter()

    rewritten_state = rewrite_query(state)
    state.update(rewritten_state)

    print_timing("Query rewriting", start)

    rewritten_question = state["rewritten_question"]

    print("\nREWRITTEN QUESTION:")
    print(rewritten_question)

    # ========================================================
    # STEP 2 — Generate Multi Queries
    # ========================================================

    print("\n\n" + "#" * 90)
    print("# STEP 2 — MULTI-QUERY GENERATION")
    print("#" * 90)

    start = time.perf_counter()

    generated_state = generate_multi_query(state)
    state.update(generated_state)

    print_timing("Multi-query generation", start)

    generated_queries = state["generated_queries"]

    print(
        f"\nNUMBER OF GENERATED QUERIES: "
        f"{len(generated_queries)}"
    )

    for index, query in enumerate(
        generated_queries,
        start=1,
    ):
        print(f"\n[{index}] {query}")

    # ========================================================
    # STEP 3 — Hybrid Retrieval
    # ========================================================

    print("\n\n" + "#" * 90)
    print("# STEP 3 — HYBRID RETRIEVAL")
    print("#" * 90)

    start = time.perf_counter()

    retrieval_state = hybrid_retrieve(state)
    state.update(retrieval_state)

    print_timing("Hybrid retrieval", start)

    retrieval_results = state["retrieval_results"]

    print(
        f"\nNumber of retrieval result groups: "
        f"{len(retrieval_results)}"
    )

    # ========================================================
    # STEP 3A — Print Dense + BM25 Results
    # ========================================================

    result_group_number = 0

    for query_index, query in enumerate(
        generated_queries,
        start=1,
    ):

        print("\n\n" + "-" * 90)
        print(f"GENERATED QUERY {query_index}")
        print("-" * 90)
        print(query)

        # ----------------------------------------------------
        # Dense results
        # ----------------------------------------------------

        dense_docs = retrieval_results[result_group_number]
        result_group_number += 1

        print(
            f"\nDENSE RETRIEVAL: "
            f"{len(dense_docs)} documents"
        )

        for rank, doc in enumerate(
            dense_docs[:TOP_K_TO_PRINT],
            start=1,
        ):
            print_document(
                f"QUERY {query_index} — DENSE",
                rank,
                doc,
            )

        # ----------------------------------------------------
        # BM25 results
        # ----------------------------------------------------

        keyword_docs = retrieval_results[result_group_number]
        result_group_number += 1

        print(
            f"\nBM25 RETRIEVAL: "
            f"{len(keyword_docs)} documents"
        )

        for rank, doc in enumerate(
            keyword_docs[:TOP_K_TO_PRINT],
            start=1,
        ):
            print_document(
                f"QUERY {query_index} — BM25",
                rank,
                doc,
            )

    # ========================================================
    # STEP 4 — Reciprocal Rank Fusion
    # ========================================================

    print("\n\n" + "#" * 90)
    print("# STEP 4 — RECIPROCAL RANK FUSION")
    print("#" * 90)

    start = time.perf_counter()

    fused_state = fuse_retrieved_documents(state)
    state.update(fused_state)

    print_timing("RRF fusion", start)

    fused_docs = state["fused_docs"]

    print(
        f"\nTotal fused documents: "
        f"{len(fused_docs)}"
    )

    print(
        f"\nTop {min(TOP_K_TO_PRINT, len(fused_docs))} "
        f"fused documents:"
    )

    for rank, doc in enumerate(
        fused_docs[:TOP_K_TO_PRINT],
        start=1,
    ):
        print_document(
            "RRF FUSED",
            rank,
            doc,
        )

    # ========================================================
    # STEP 5 — Context Compression / Selection
    # ========================================================

    print("\n\n" + "#" * 90)
    print("# STEP 5 — FINAL CONTEXT SELECTION")
    print("#" * 90)

    start = time.perf_counter()

    compressed_state = compress_context(state)
    state.update(compressed_state)

    print_timing("Context compression/selection", start)

    compressed_docs = state["compressed_docs"]

    print(
        f"\nFINAL_CONTEXT_DOCUMENTS = "
        f"{FINAL_CONTEXT_DOCUMENTS}"
    )

    print(
        f"Documents passed to answer generation: "
        f"{len(compressed_docs)}"
    )

    for rank, doc in enumerate(
        compressed_docs,
        start=1,
    ):
        print_document(
            "FINAL CONTEXT",
            rank,
            doc,
        )

    # ========================================================
    # STEP 6 — Final Formatted Context
    # ========================================================

    print("\n\n" + "#" * 90)
    print("# STEP 6 — FORMATTED CONTEXT SENT TO LLM")
    print("#" * 90)

    final_context = format_context(compressed_docs)

    print("\n")
    print(final_context)

    # ========================================================
    # Final Summary
    # ========================================================

    print("\n\n" + "#" * 90)
    print("# DEBUG SUMMARY")
    print("#" * 90)

    print("\nOriginal question:")
    print(QUERY)

    print("\nRewritten question:")
    print(rewritten_question)

    print(
        f"\nGenerated queries: "
        f"{len(generated_queries)}"
    )

    print(
        f"Retrieved result groups: "
        f"{len(retrieval_results)}"
    )

    print(
        f"Fused documents: "
        f"{len(fused_docs)}"
    )

    print(
        f"Final context documents: "
        f"{len(compressed_docs)}"
    )

    print("\n" + "#" * 90)
    print("# DEBUG COMPLETE")
    print("#" * 90)


if __name__ == "__main__":
    main()