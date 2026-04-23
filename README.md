# Production RAG Pipeline

A production-grade Retrieval Augmented Generation system built with Pinecone, LangChain, and Groq.

## Stack
- **Vector DB:** Pinecone (cloud, serverless)
- **Embeddings:** BAAI/bge-base-en-v1.5 (HuggingFace, local)
- **LLM:** Groq (llama-3.1-8b-instant)
- **Chunking:** LangChain RecursiveCharacterTextSplitter
- **Backend:** FastAPI (coming soon)

## Architecture