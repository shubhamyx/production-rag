# Production RAG Pipeline

A production-grade Retrieval Augmented Generation system built with Pinecone, LangChain, and Groq.

## Stack
- **Vector DB:** Pinecone (cloud, serverless)
- **Embeddings:** BAAI/bge-base-en-v1.5 (HuggingFace, local)
- **LLM:** Groq (llama-3.1-8b-instant)
- **Chunking:** LangChain RecursiveCharacterTextSplitter
- **Backend:** FastAPI (coming soon)

## Architecture

Document → Chunking → BGE Embeddings → Pinecone Upsert
Query → BGE Embeddings → Pinecone Retrieval → Groq LLM → Answer

## Project Structure
production-rag/
├── config.py          # All configuration and env vars
├── embeddings.py      # BGE embedding model wrapper
├── ingest.py          # Document chunking and Pinecone upsert
├── retriever.py       # Semantic search from Pinecone
├── main.py            # Full RAG query pipeline
└── .env               # API keys (not committed)

## Setup

1. Clone the repo
2. Install dependencies:
```bash
pip install pinecone langchain langchain-text-splitters sentence-transformers groq python-dotenv fastapi uvicorn
```
3. Create `.env`:
PINECONE_API_KEY=your_key
GROQ_API_KEY=your_key
4. Run:
```bash
python main.py
```

## Part of My 30-Day AI Developer Challenge
Building in public — follow along: [@shubhamyx](https://github.com/shubhamyx)