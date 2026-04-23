from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

model = SentenceTransformer(EMBEDDING_MODEL)

def get_embeddings(texts: list[str]) -> list[list[float]]:
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()