from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os

from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStoreManager
from app.services.rag_engine import RAGEngine

app = FastAPI(title="Production RAG API", version="1.0.0")

doc_processor = DocumentProcessor()
vector_store = VectorStoreManager()
rag_engine = RAGEngine(vector_store)

class QueryRequest(BaseModel):
    query: str

@app.post("/api/v1/ingest")
async def ingest_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported currently.")
    
    upload_dir = "./data/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Ingestion Flow
    docs = doc_processor.extract_text_with_metadata(file_path, file.filename)
    vector_store.add_documents(docs)
    
    return {
        "status": "Success",
        "filename": file.filename,
        "chunks_indexed": len(docs)
    }

@app.post("/api/v1/chat")
async def chat_endpoint(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    result = rag_engine.generate_response(request.query)
    return result