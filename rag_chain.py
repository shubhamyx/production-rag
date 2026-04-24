from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from retriever import retrieve
from config import GROQ_API_KEY

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0.1
)

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question based ONLY on the context below.
If the answer is not in the context, say "I don't know based on the provided context."

Context:
{context}

Question:
{question}

Answer:
""")

def format_context(inputs):
    results = retrieve(
        query=inputs["question"],
        namespace=inputs.get("namespace", "docs"),
        filter_by=inputs.get("filter_by", None)
    )
    return "\n\n".join([r["text"] for r in results])

rag_chain = (
    RunnablePassthrough.assign(context=format_context)
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    result = rag_chain.invoke({
        "question": "what is langchain?",
        "namespace": "docs"
    })
    print(result)