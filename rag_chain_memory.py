from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from retriever import retrieve
from config import GROQ_API_KEY

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0.1
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer based ONLY on the context below.
Context:
{context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

chat_history = []

def format_context(inputs):
    results = retrieve(
        query=inputs["question"],
        namespace=inputs.get("namespace", "docs")
    )
    return "\n\n".join([r["text"] for r in results])

rag_chain = (
    RunnablePassthrough.assign(context=format_context)
    | prompt
    | llm
    | StrOutputParser()
)

def chat(question: str, namespace: str = "docs"):
    global chat_history
    
    result = rag_chain.invoke({
        "question": question,
        "namespace": namespace,
        "history": chat_history
    })
    
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=result))
    
    return result

if __name__ == "__main__":
    print("Q1:", chat("what is langchain?"))
    print("---")
    print("Q2:", chat("what are its main components?"))
    print("---")
    print("Q3:", chat("can you summarize what we just discussed?"))