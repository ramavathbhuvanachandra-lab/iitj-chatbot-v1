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

print("=" * 80)
print(f"Loaded Documents : {len(documents)}")
print(f"Created Chunks   : {len(chunks)}")
print("=" * 80)


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

    docs = retriever.invoke(query)

    print("\n" + "-" * 80)
    print("DENSE RETRIEVAL")
    print("-" * 80)
    print(f"Query : {query}")
    print()

    for index, doc in enumerate(docs, start=1):

        source = doc.metadata.get("source", "Unknown")

        print(f"[Dense {index}]")
        print(f"Source : {source}")
        print(doc.page_content[:200])
        print()

    return docs


# =========================================================
# Keyword Retrieval
# =========================================================

def keyword_retrieve(query: str):

    docs = bm25_retriever.invoke(query)

    print("\n" + "-" * 80)
    print("BM25 RETRIEVAL")
    print("-" * 80)
    print(f"Query : {query}")
    print()

    for index, doc in enumerate(docs, start=1):

        source = doc.metadata.get("source", "Unknown")

        print(f"[BM25 {index}]")
        print(f"Source : {source}")
        print(doc.page_content[:200])
        print()

    return docs


# =========================================================
# Reciprocal Rank Fusion
# =========================================================

RRF_K = 60


def get_document_id(document):

    metadata = document.metadata

    source = metadata.get("source", "")
    page = metadata.get("page", "")

    if source:
        return f"{source}_{page}"

    return hash(document.page_content)


def reciprocal_rank_fusion(ranked_lists, k=RRF_K):

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

    print("\n" + "=" * 80)
    print("RRF RESULTS")
    print("=" * 80)

    for index, (doc_id, score) in enumerate(fused_documents, start=1):

        doc = document_lookup[doc_id]

        source = doc.metadata.get("source", "Unknown")

        print(f"{index}. Score : {score:.5f}")
        print(f"   Source : {source}")
        print()

    print("=" * 80)

    return [
        document_lookup[document_id]
        for document_id, _ in fused_documents
    ]


# =========================================================
# Final Context
# =========================================================

FINAL_CONTEXT_DOCUMENTS = 5


def format_context(documents):

    formatted_context = []

    print("\n" + "=" * 80)
    print("FINAL DOCUMENTS SENT TO LLM")
    print("=" * 80)

    for index, document in enumerate(documents, start=1):

        source = document.metadata.get("source", "Unknown")

        print(f"[Document {index}]")
        print(f"Source : {source}")
        print(document.page_content[:300])
        print()

        formatted_context.append(
            f"Document {index}\n"
            f"{document.page_content}"
        )

    print("=" * 80)

    return "\n\n".join(formatted_context)