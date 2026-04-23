from main import rag_query

result = rag_query("what is langchain and what are its main features?", namespace="docs")
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
print(f"Chunks retrieved: {result['chunks_retrieved']}")