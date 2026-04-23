from pinecone import Pinecone
from embeddings import get_embeddings
from config import PINECONE_API_KEY, INDEX_NAME

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

query_embedding = get_embeddings(["langchain"])[0]

results = index.query(
    vector=query_embedding,
    top_k=1,
    namespace="docs",
    include_metadata=True
)

print(results)