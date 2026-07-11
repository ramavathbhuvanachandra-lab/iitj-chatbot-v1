from backend.state import GraphState
from backend.llm import llm
from typing import List

from backend.prompts import (
    rewrite_prompt,
    multi_query_prompt,
    answer_prompt,
)
from backend.retriever import (
    dense_retrieve,
    keyword_retrieve,
    reciprocal_rank_fusion,
    FINAL_CONTEXT_DOCUMENTS,
    format_context,
)


rewrite_chain = rewrite_prompt | llm


def rewrite_query(state: GraphState) -> GraphState:
    """
    Rewrite the user's question into a retrieval-friendly query.
    """

    rewritten_question = rewrite_chain.invoke(
        {
            "question": state["question"],
            "chat_history": state["chat_history"]
        }
    ).content.strip()

    return {
        "rewritten_question": rewritten_question
    }

multi_query_chain = multi_query_prompt | llm

answer_chain = answer_prompt | llm

def parse_generated_queries(response: str) -> list[str]:
    """
    Convert the LLM response into a list of search queries.
    """

    queries = []

    for line in response.split("\n"):
        line = line.strip()

        if line:
            queries.append(line)

    return queries

def generate_multi_query(state: GraphState) -> GraphState:
    """
    Generate multiple semantic search queries from the rewritten question.
    """

    response = multi_query_chain.invoke(
        {
            "rewritten_question": state["rewritten_question"]
        }
    )

    generated_queries = parse_generated_queries(
        response.content
    )

    return {
        "generated_queries": generated_queries
    }

def hybrid_retrieve(state: GraphState) -> GraphState:
    """
    Retrieve documents using both dense and BM25 retrieval.
    """

    retrieval_results = []

    for query in state["generated_queries"]:

        retrieval_results.append(
            dense_retrieve(query)
        )

        retrieval_results.append(
            keyword_retrieve(query)
        )

    return {
        "retrieval_results": retrieval_results
    }



def fuse_retrieved_documents(state: GraphState) -> GraphState:
    """
    Fuse dense and keyword retrieval results using Reciprocal Rank Fusion.
    """

    fused_docs = reciprocal_rank_fusion(
        state["retrieval_results"]
    )

    return {
        "fused_docs": fused_docs
    }

def compress_context(state: GraphState) -> GraphState:
    """
    Keep only the top-k fused documents for answer generation.
    """

    compressed_docs = state["fused_docs"][:FINAL_CONTEXT_DOCUMENTS]

    return {
        "compressed_docs": compressed_docs
    }



def generate_answer(state: GraphState) -> GraphState:
    """
    Generate the final answer using the retrieved context.
    """

    context = format_context(
        state["compressed_docs"]
    )

    response = answer_chain.invoke(
        {
            "context": context,
            "question": state["question"],
            "chat_history": state["chat_history"]
        }
    )

    return {
        "answer": response.content.strip()
    }