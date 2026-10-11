"""
vectordb.py — turns a PDF into a searchable FAISS vector database.

    PDF  ->  pages  ->  chunks  ->  embeddings (numbers)  ->  FAISS
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config

# Loaded ONCE when the server starts (loading the model takes several seconds)
embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


def build_vector_db(pdf_path: str) -> FAISS:
    pages = PyPDFLoader(pdf_path).load()
    print(f"[STEP 1] Loaded {len(pages)} pages from {pdf_path}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    chunks = splitter.split_documents(pages)
    print(f"[STEP 2] Split the text into {len(chunks)} chunks")

    vector_db = FAISS.from_documents(chunks, embeddings)
    print(f"[STEP 3] Stored {vector_db.index.ntotal} vectors (each {vector_db.index.d} numbers long) in FAISS")
    return vector_db
