"""
main.py — the FastAPI server: it only handles HTTP and calls the other modules.

Run (from the backend/ folder):   uvicorn app.main:app --reload --port 8000
Docs:                             http://localhost:8000/docs
"""

import os
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.agent import ask, build_agent
from app.schemas import QueryRequest, QueryResponse
from app.vectordb import build_vector_db

app = FastAPI(title="PDF RAG")

# Allow the Streamlit app (port 8501) to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(config.UPLOAD_DIR, exist_ok=True)
vector_db = None      # the FAISS database lives in memory until the server stops
document_name = None  # name of the uploaded file, so the agent knows what it is reading


@app.get("/status")
def status():
    """Which document is loaded right now? (The chat app shows this in its sidebar.)"""
    if vector_db is None:
        return {"document": None, "chunks": 0, "vector_size": 0}
    return {"document": document_name, "chunks": vector_db.index.ntotal, "vector_size": vector_db.index.d}


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    global vector_db, document_name
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file")

    path = os.path.join(config.UPLOAD_DIR, os.path.basename(file.filename))
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)  # save the uploaded bytes to disk

    vector_db = build_vector_db(path)
    document_name = file.filename
    return {"filename": file.filename, "chunks": vector_db.index.ntotal, "vector_size": vector_db.index.d}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if vector_db is None:
        raise HTTPException(status_code=400, detail="Upload a PDF first")
    try:
        agent, found_chunks = build_agent(request.provider, vector_db, document_name)
        answer = ask(agent, request.question, request.session_id)
    except Exception as error:  # e.g. missing/wrong API key, rate limit
        raise HTTPException(status_code=500, detail=f"{type(error).__name__}: {error}")

    return QueryResponse(answer=answer, sources=found_chunks)
