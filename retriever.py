from pinecone import Pinecone
from embeddings import get_embeddings
from config import PINECONE_API_KEY, INDEX_NAME, TOP_K

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

def retrieve(query: str, namespace: str = "default", top_k: int = TOP_K, filter_by: dict = None) :
    query_embedding = get_embeddings([query])[0]
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
        filter=filter_by
    )
    
    contexts = []
    for match in results["matches"]:
        contexts.append({
            "text": match["metadata"]["text"],
            "source": match["metadata"]["source"],
            "score": match["score"]
        })
    
    return contexts