from collections import defaultdict

from langchain_community.retrievers import BM25Retriever

from backend.config import DATA_PATH
from backend.ingestion import load_documents, split_documents
from backend.vectorstore import vectorstore


# =========================================================
# Load Documents
# =========================================================

documents = load_documents(DATA_PATH)
chunks = split_documents(documents)


# =========================================================
# BM25 Retriever
# =========================================================

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5


# =========================================================
# Dense Retriever
# =========================================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)


# =========================================================
# Dense Retrieval
# =========================================================

def dense_retrieve(query: str):
    """
    Retrieve the top-k documents using dense vector similarity search.
    """

    return retriever.invoke(query)


# =========================================================
# Keyword Retrieval
# =========================================================

def keyword_retrieve(query: str):
    """
    Retrieve the top-k documents using BM25 keyword search.
    """

    return bm25_retriever.invoke(query)


# =========================================================
# Reciprocal Rank Fusion
# =========================================================

RRF_K = 60


def get_document_id(document):
    """
    Generate a unique identifier for each retrieved document.
    """

    source = document.metadata.get("source", "")

    return f"{source}_{hash(document.page_content)}"


def reciprocal_rank_fusion(ranked_lists, k=RRF_K):
    """
    Combine multiple ranked retrieval results using
    Reciprocal Rank Fusion (RRF).
    """

    document_scores = defaultdict(float)
    document_lookup = {}

    for ranked_documents in ranked_lists:

        for rank, document in enumerate(ranked_documents):

            document_id = get_document_id(document)

            document_lookup[document_id] = document

            document_scores[document_id] += 1 / (k + rank + 1)

    fused_documents = sorted(
        document_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        document_lookup[document_id]
        for document_id, _ in fused_documents
    ]


# =========================================================
# Final Context
# =========================================================

FINAL_CONTEXT_DOCUMENTS = 10


def format_context(documents):
    """
    Format retrieved documents into a single context string
    that will be provided to the LLM.
    """

    formatted_context = []

    for index, document in enumerate(documents, start=1):

        formatted_context.append(
            f"Document {index}\n"
            f"{document.page_content}"
        )

    return "\n\n".join(formatted_context)