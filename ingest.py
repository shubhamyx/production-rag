from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embeddings import get_embeddings
from config import PINECONE_API_KEY, INDEX_NAME, CHUNK_SIZE, CHUNK_OVERLAP
import uuid

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

def ingest_documents(texts: list[str], source: str, namespace: str = "default"):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )

    all_vectors = []

    for text in texts:
        chunks = splitter.split_text(text)
        embeddings = get_embeddings(chunks)

        for chunk, embedding in zip(chunks, embeddings):
            all_vectors.append({
                "id": str(uuid.uuid4()),
                "values": embedding,
                "metadata": {
                    "text": chunk,
                    "source": source,
                    "namespace": namespace
                }
            })

    index.upsert(vectors=all_vectors, namespace=namespace)
    print(f"Ingested {len(all_vectors)} chunks from source: {source}")
    return len(all_vectors)