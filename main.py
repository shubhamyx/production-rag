from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from retriever import retrieve
from ingest import ingest_documents
from groq import Groq
from config import GROQ_API_KEY
from pypdf import PdfReader
from retriever import retrieve    
import tempfile
import os
from pinecone import Pinecone as PineconeClient
from config import PINECONE_API_KEY, INDEX_NAME

app = FastAPI(title="Production RAG API", version="1.0.0")

groq_client = Groq(api_key=GROQ_API_KEY)

_pc = PineconeClient(api_key=PINECONE_API_KEY)
_index = _pc.Index(INDEX_NAME)


class QueryRequest(BaseModel):
    question: str
    namespace: str = "default"
    category: str = None
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_retrieved: int

@app.get("/")
def root():
    return {"message": "Production RAG API is running"}

@app.get("/stats")
def get_stats():
    try:
        stats = _index.describe_index_stats()
        namespaces = {k: {"vector_count": v["vector_count"]} for k, v in stats["namespaces"].items()}
        return {
            "total_vectors": int(stats["total_vector_count"]),
            "namespaces": namespaces,
            "dimension": int(stats["dimension"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/ingest")
async def ingest_pdf(
    file: UploadFile = File(...),
    namespace: str = "default",
    category: str = "general"
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        reader = PdfReader(tmp_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"
        
        count = ingest_documents(
            texts=[full_text],
            source=file.filename,
            namespace=namespace,
            category=category,
            date="2026"
        )
        
        return {
            "message": "Ingestion successful",
            "filename": file.filename,
            "chunks_ingested": count,
            "namespace": namespace,
            "category": category
        }
    finally:
        os.unlink(tmp_path)

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    filter_by = None
    if request.category:
        filter_by = {"category": {"$eq": request.category}}
    
    contexts = retrieve(
        query=request.question,
        namespace=request.namespace,
        top_k=request.top_k,
        filter_by=filter_by
    )
    
    if not contexts:
        return QueryResponse(
            answer="No relevant context found.",
            sources=[],
            chunks_retrieved=0
        )
    
    context_text = "\n\n".join([c["text"] for c in contexts])
    sources = list(set([c["source"] for c in contexts]))
    
    prompt = f"""Answer the question based ONLY on the context below.
If the answer is not in the context, say "I don't know based on the provided context."

Context:
{context_text}

Question: {request.question}

Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    
    return QueryResponse(
        answer=response.choices[0].message.content,
        sources=sources,
        chunks_retrieved=len(contexts)
    )