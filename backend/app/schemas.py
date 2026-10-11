"""
schemas.py — the shape of the JSON that goes in and out of the API (Pydantic models).
FastAPI uses these to validate requests and to build the /docs page.
"""

from pydantic import BaseModel


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
