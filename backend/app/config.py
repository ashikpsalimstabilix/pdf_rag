"""
config.py — every setting in ONE place.

Production rule: no file except this one should read environment variables.
Values come from the .env file (secrets) or fall back to the defaults below.
"""

import os

from dotenv import load_dotenv

load_dotenv()  # reads GEMINI_API_KEY / GROQ_API_KEY from the .env file

# ---- LLM ----
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")  # Groq retires models often: llama3-8b-8192 and llama-3.1-8b-instant are gone

# ---- Embeddings + chunking ----
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # runs locally, no API key
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# ---- Retrieval ----
TOP_K = 4  # how many chunks the search tool returns

# ---- Server ----
UPLOAD_DIR = "uploads"
ALLOWED_ORIGINS = ["http://localhost:8501", "http://127.0.0.1:8501"]
