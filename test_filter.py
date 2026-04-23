from retriever import retrieve

# only retrieve from "langchain" category
results = retrieve(
    "what is langchain?",
    namespace="docs",
    filter_by={"category": {"$eq": "langchain"}}
)
print(f"Results with filter: {len(results)}")