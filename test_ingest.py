from ingest import ingest_documents

sample = ["""
Artificial intelligence is transforming industries worldwide.
Machine learning models are being deployed in healthcare, finance, and education.
Large language models have revolutionized natural language processing tasks.
RAG systems combine retrieval with generation for more accurate responses.
"""]

count = ingest_documents(sample, source="test-doc", namespace="test")
print(f"Total chunks ingested: {count}")