from sentence_transformers import CrossEncoder
from retriever import retrieve

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def retrieve_and_rerank(query: str, namespace: str = "docs", top_k_retrieve: int = 10, top_k_final: int = 3):
    # Stage 1: retrieve more chunks than needed
    candidates = retrieve(query=query, namespace=namespace, top_k=top_k_retrieve)
    
    if not candidates:
        return []
    
    # Stage 2: rerank using cross-encoder
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)
    
    # Sort by reranker score
    ranked = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True
    )
    
    # Return top_k_final with reranker scores
    results = []
    for score, candidate in ranked[:top_k_final]:
        candidate["reranker_score"] = float(score)
        results.append(candidate)
    
    return results

if __name__ == "__main__":
    results = retrieve_and_rerank("what are langchain agents?", namespace="docs")
    for r in results:
        print(f"Reranker Score: {r['reranker_score']:.4f}")
        print(f"Vector Score: {r['score']:.4f}")
        print(f"Text: {r['text'][:150]}")
        print("---")