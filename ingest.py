from backend.ingestion import (
    load_documents,
    split_documents,
    create_vectorstore,
)
from backend.config import DATA_PATH

print("=" * 60)
print("Loading Documents...")
print("=" * 60)

documents = load_documents(DATA_PATH)

print(f"Loaded {len(documents)} documents.")

print("=" * 60)
print("Splitting Documents...")
print("=" * 60)

chunks = split_documents(documents)

print(f"Created {len(chunks)} chunks.")

print("=" * 60)
print("Building Chroma Vector Database...")
print("=" * 60)

vectorstore = create_vectorstore(chunks)

print(f"Stored {vectorstore._collection.count()} vectors.")

print("=" * 60)
print("Vector Database Ready!")
print("=" * 60)