from langchain_community.retrievers import BM25Retriever

from backend.ingestion import load_documents, split_documents
from backend.vectorstore import vectorstore
from backend.config import DATA_PATH

documents = load_documents(DATA_PATH)

# Build BM25 index
documents = load_documents("/Users/bhuvanachandra/Desktop/data")
chunks = split_documents(documents)

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5


# -------------------------
# Dense Retrieval
# -------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)


def dense_retrieve(query: str):
    return retriever.invoke(query)

# -------------------------
# Keyword Retrieval
# -------------------------

def keyword_retrieve(query: str):
    return bm25_retriever.invoke(query)

from collections import defaultdict

RRF_K = 60


def get_document_id(document):
    """
    Generate a unique identifier for a retrieved document.
    """

    metadata = document.metadata

    source = metadata.get("source", "")
    page = metadata.get("page", "")

    if source:
        return f"{source}_{page}"

    return hash(document.page_content)


def reciprocal_rank_fusion(ranked_lists, k=RRF_K):
    """
    Merge multiple ranked retrieval results using Reciprocal Rank Fusion (RRF).
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


FINAL_CONTEXT_DOCUMENTS = 5

def format_context(documents):
    """
    Convert retrieved documents into a single formatted context string.
    """

    formatted_context = []

    for index, document in enumerate(documents, start=1):

        formatted_context.append(
            f"Document {index}\n"
            f"{document.page_content}"
        )

    return "\n\n".join(formatted_context)