from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_engine import HybridRAGEngine
import uuid

app = FastAPI(title="Local-First Hybrid RAG Knowledge Base API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_engine = HybridRAGEngine()

class DocumentInput(BaseModel):
    text: str
    title: str = "Untitled Document"

class QueryInput(BaseModel):
    query: str

@app.post("/api/add-document")
def add_document_endpoint(doc: DocumentInput):
    if not doc.text.strip():
        raise HTTPException(status_code=400, detail="Document text cannot be empty")
    
    doc_id = str(uuid.uuid4())
    rag_engine.add_document(doc_id=doc_id, text=doc.text, metadata={"title": doc.title})
    return {"status": "success", "doc_id": doc_id, "message": "Document added and indexed locally successfully."}

@app.post("/api/query")
def query_endpoint(q: QueryInput):
    if not q.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    result = rag_engine.generate_answer(q.query)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
