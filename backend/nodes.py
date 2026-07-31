from backend.state import GraphState
from backend.llm import llm
import time
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

import time
# =========================================================
# Chains
# =========================================================

rewrite_chain = rewrite_prompt | llm
multi_query_chain = multi_query_prompt | llm
answer_chain = answer_prompt | llm


# =========================================================
# Rewrite Query
# =========================================================
def rewrite_query(state: GraphState) -> GraphState:

   

    rewritten_question = rewrite_chain.invoke(
        {
            "question": state["question"],
            "chat_history": state["chat_history"]
        }
    ).content.strip()


  

    return {
        "rewritten_question": rewritten_question
    }

# =========================================================
# Parse Generated Queries
# =========================================================

def parse_generated_queries(response: str) -> list[str]:

    queries = []

    for line in response.split("\n"):

        line = line.strip()

        if line:
            queries.append(line)

    return queries


# =========================================================
# Generate Multi Query
# =========================================================

def generate_multi_query(state: GraphState) -> GraphState:

   

    response = multi_query_chain.invoke(
        {
            "rewritten_question": state["rewritten_question"]
        }
    )

    generated_queries = parse_generated_queries(
        response.content
    )

    

   
    for index, query in enumerate(generated_queries, start=1):
        pass

   

    return {
        "generated_queries": generated_queries
    }

# =========================================================
# Hybrid Retrieval
# =========================================================

def hybrid_retrieve(state: GraphState) -> GraphState:

    

    retrieval_results = []

    
    for index, query in enumerate(state["generated_queries"], start=1):

       

        dense_docs = dense_retrieve(query)
        keyword_docs = keyword_retrieve(query)

        retrieval_results.append(dense_docs)
        retrieval_results.append(keyword_docs)


    return {
        "retrieval_results": retrieval_results
    }
# =========================================================
# Fuse Documents
# =========================================================

def fuse_retrieved_documents(state: GraphState) -> GraphState:

  

    fused_docs = reciprocal_rank_fusion(
        state["retrieval_results"]
    )

 

    return {
        "fused_docs": fused_docs
    }
# =========================================================
# Compress Context
# =========================================================

def compress_context(state: GraphState) -> GraphState:

   

    compressed_docs = state["fused_docs"][:FINAL_CONTEXT_DOCUMENTS]

    

    return {
        "compressed_docs": compressed_docs
    }

# =========================================================
# Generate Answer
# =========================================================

def generate_answer(state: GraphState) -> GraphState:

    

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
        "answer": response.content.strip(),
        "context": context,
    }