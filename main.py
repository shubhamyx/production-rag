from retriever import retrieve
from groq import Groq
from config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)

def rag_query(question: str, namespace: str = "default") -> dict:
    contexts = retrieve(question, namespace=namespace)
    
    if not contexts:
        return {"answer": "No relevant context found.", "sources": []}
    
    context_text = "\n\n".join([c["text"] for c in contexts])
    sources = list(set([c["source"] for c in contexts]))
    
    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the context below.
If the answer is not in the context, say "I don't know based on the provided context."

Context:
{context_text}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    
    return {
        "answer": response.choices[0].message.content,
        "sources": sources,
        "chunks_retrieved": len(contexts)
    }

if __name__ == "__main__":
    result = rag_query("what is machine learning?", namespace="test")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    print(f"Chunks retrieved: {result['chunks_retrieved']}")