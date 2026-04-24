from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from retriever import retrieve
from config import GROQ_API_KEY
from datetime import datetime

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0.1
)

@tool
def search_documents(query: str) -> str:
    """Search the knowledge base for information about LangChain or AI."""
    results = retrieve(query=query, namespace="docs")
    if not results:
        return "No relevant documents found."
    return "\n\n".join([r["text"] for r in results[:2]])

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression like '2 + 2' or '10 * 5'."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

def get_current_date() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [search_documents, calculator] 



agent = create_agent(llm, tools)

def run_agent(question: str):
    result = agent.invoke({
        "messages": [HumanMessage(content=question)]
    })
    return result["messages"][-1].content

if __name__ == "__main__":
    print("\n--- Test 1: Document Search ---")
    print(run_agent("what is langchain?"))

    print("\n--- Test 2: Calculator ---")
    print(run_agent("what is 1234 * 5678?"))
