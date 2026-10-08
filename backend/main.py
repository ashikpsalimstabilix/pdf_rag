"""
main.py — a small FastAPI server around core_rag.py

Run:   uvicorn main:app --reload --port 8000
Docs:  http://localhost:8000/docs   (try the endpoints from your browser!)
"""

import os
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core_rag import ask, build_agent, build_vector_db

app = FastAPI(title="PDF RAG Workshop API")

# Allow the Streamlit app (port 8501) to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
vector_db = None  # the FAISS database lives in memory until the server stops


class QueryRequest(BaseModel):
    question: str
    provider: str = "groq"        # "groq" or "gemini"
    session_id: str = "default"   # same id = the agent remembers the conversation


class Source(BaseModel):
    query: str
    text: str
    page: int
    distance: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    global vector_db
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file")

    path = os.path.join(UPLOAD_DIR, os.path.basename(file.filename))
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)  # save the uploaded bytes to disk

    vector_db = build_vector_db(path)
    return {"filename": file.filename, "chunks": vector_db.index.ntotal, "vector_size": vector_db.index.d}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if vector_db is None:
        raise HTTPException(status_code=400, detail="Upload a PDF first")
    try:
        agent, found_chunks = build_agent(request.provider, vector_db)
        answer = ask(agent, request.question, request.session_id)
    except Exception as error:  # e.g. missing/wrong API key, rate limit
        raise HTTPException(status_code=500, detail=f"{type(error).__name__}: {error}")

    return QueryResponse(answer=answer, sources=found_chunks)
