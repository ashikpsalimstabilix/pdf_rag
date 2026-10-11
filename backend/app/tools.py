"""
tools.py — the functions the agent is allowed to call.

A tool is a normal Python function. The docstring is IMPORTANT: it is how the
LLM learns what the tool does and when to use it.
"""

from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool

from app import config


def make_search_tool(vector_db: FAISS):
    """Create a search tool bound to one document's vector database.

    Returns (tool, found_chunks). The tool writes every chunk it finds into
    `found_chunks`, so the web UI can show what was retrieved.
    """
    found_chunks = []

    @tool
    def search_pdf(query: str) -> str:
        """Search the uploaded document and return the most relevant passages (with page numbers).
        Use short keyword queries, e.g. "engine power", "price", "warranty", "exam duration"."""
        results = vector_db.similarity_search_with_score(query, k=config.TOP_K)
        print(f"[TOOL] search_pdf({query!r}) -> distances {[round(float(s), 3) for _, s in results]}")
        for doc, score in results:
            found_chunks.append({
                "query": query,
                "text": doc.page_content,
                "page": doc.metadata.get("page", 0) + 1,
                "distance": round(float(score), 4),
            })
        return "\n\n".join(f"[page {doc.metadata.get('page', 0) + 1}]\n{doc.page_content}" for doc, _ in results)

    return search_pdf, found_chunks
