# Production RAG Pipeline

A production-grade Retrieval Augmented Generation system built with Pinecone, LangChain, and Groq.

## Stack
- **Vector DB:** Pinecone (cloud, serverless, 768 dims)
- **Embeddings:** BAAI/bge-base-en-v1.5 (HuggingFace, runs locally)
- **LLM:** Groq (llama-3.1-8b-instant)
- **Chunking:** LangChain RecursiveCharacterTextSplitter
- **API:** FastAPI

## Architecture
Document → Chunking → BGE Embeddings → Pinecone Upsert
Query → BGE Embeddings → Pinecone Retrieval → Groq LLM → Answer
## Project Structure
production-rag/
├── config.py          # Configuration and env vars
├── embeddings.py      # BGE embedding model wrapper
├── ingest.py          # Document chunking and Pinecone upsert
├── retriever.py       # Semantic search from Pinecone
├── main.py            # FastAPI app
└── .env               # API keys (not committed)
## Setup

1. Clone the repo
2. Install dependencies:
```bash
pip install pinecone langchain langchain-text-splitters sentence-transformers groq python-dotenv fastapi uvicorn pypdf
```
3. Create `.env`:
PINECONE_API_KEY=your_pinecone_key
GROQ_API_KEY=your_groq_key
4. Start the API:
```bash
uvicorn main:app --reload
```

## API Endpoints

### GET /stats
Returns index statistics.
```json
{
  "total_vectors": 72,
  "namespaces": {
    "docs": {"vector_count": 71}
  },
  "dimension": 768
}
```

### POST /ingest
Upload a PDF and ingest it into Pinecone.
- **file**: PDF file (multipart/form-data)
- **namespace**: where to store (default: "default")
- **category**: tag for metadata filtering (default: "general")

Response:
```json
{
  "message": "Ingestion successful",
  "filename": "document.pdf",
  "chunks_ingested": 71,
  "namespace": "docs",
  "category": "langchain"
}
```

### POST /query
Query the RAG pipeline with optional metadata filtering.

Request:
```json
{
  "question": "what is langchain?",
  "namespace": "docs",
  "category": "langchain",
  "top_k": 3
}
```

Response:
```json
{
  "answer": "LangChain is an open-source framework...",
  "sources": ["document.pdf"],
  "chunks_retrieved": 3
}
```

## Part of My 30-Day AI Developer Challenge
Building in public — follow along: [@shubhamyx](https://github.com/shubhamyx)
