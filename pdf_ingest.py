from pypdf import PdfReader
from ingest import ingest_documents

def ingest_pdf(pdf_path: str, namespace: str = "default", category: str = "general"):
    reader = PdfReader(pdf_path)
    
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    
    print(f"Extracted {len(reader.pages)} pages from {pdf_path}")
    
    count = ingest_documents(
        texts=[full_text],
        source=pdf_path,
        namespace=namespace,
        category="langchain",
        date="2026"
    )
    
    return count

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_ingest.py <path_to_pdf>")
    else:
        count = ingest_pdf(sys.argv[1], namespace="docs")
        print(f"Done. {count} chunks ingested.")