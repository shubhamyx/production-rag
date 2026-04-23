from retriever import retrieve

results = retrieve("what is langchain?", namespace="docs")
for r in results:
    print(f"Score: {r['score']:.4f}")
    print(f"Text: {r['text'][:150]}")
    print("---")