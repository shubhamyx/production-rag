from retriever import retrieve

results = retrieve("what is machine learning?", namespace="test")
for r in results:
    print(f"Score: {r['score']:.4f}")
    print(f"Source: {r['source']}")
    print(f"Text: {r['text'][:100]}")
    print("---")