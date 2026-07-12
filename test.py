from backend.vectorstore import vectorstore

print("=" * 60)
print("NUMBER OF VECTORS")
print("=" * 60)

print(vectorstore._collection.count())