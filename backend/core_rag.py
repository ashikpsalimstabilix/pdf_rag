"""
core_rag.py — the "brain" of our app (the same ideas as the two notebooks).

  1. build_vector_db(pdf_path)     PDF -> chunks -> vectors -> FAISS
  2. build_agent(provider, db)     an LLM agent (LangChain + LangGraph) that:
                                     - follows a SYSTEM PROMPT
                                     - can call a "search_pdf" TOOL
                                     - has MEMORY of the conversation
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()  # reads GEMINI_API_KEY / GROQ_API_KEY from the .env file

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  # the old "llama3-8b-8192" has been retired by Groq

SYSTEM_PROMPT = """You are a friendly teaching assistant for engineering students.
Always use the search_pdf tool to look up facts in the uploaded PDF before answering.
Answer ONLY from what the tool returns. If the PDF does not contain the answer, say so honestly.
Keep answers short and clear."""

# Memory: remembers every conversation (one "thread" per chat session) while the server runs
memory = InMemorySaver()


# ---------------------------------------------------------------------------
# Part 1: PDF -> FAISS vector database
# ---------------------------------------------------------------------------
def build_vector_db(pdf_path: str) -> FAISS:
    pages = PyPDFLoader(pdf_path).load()
    print(f"[STEP 1] Loaded {len(pages)} pages from {pdf_path}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(pages)
    print(f"[STEP 2] Split the text into {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # runs locally, no API key
    vector_db = FAISS.from_documents(chunks, embeddings)
    print(f"[STEP 3] Stored {vector_db.index.ntotal} vectors (each {vector_db.index.d} numbers long) in FAISS")
    return vector_db


# ---------------------------------------------------------------------------
# Part 2: choose an LLM (provider = "gemini" or "groq")
# ---------------------------------------------------------------------------
def get_llm(provider: str):
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0)

    from langchain_groq import ChatGroq
    return ChatGroq(model=GROQ_MODEL, api_key=os.getenv("GROQ_API_KEY"), temperature=0)


# ---------------------------------------------------------------------------
# Part 3: the agent = LLM + system prompt + tool + memory
# ---------------------------------------------------------------------------
def build_agent(provider: str, vector_db: FAISS):
    found_chunks = []  # the tool writes here, so the UI can show what was retrieved

    @tool
    def search_pdf(query: str) -> str:
        """Search the uploaded PDF and return the most relevant passages."""
        results = vector_db.similarity_search_with_score(query, k=4)
        print(f"[TOOL] search_pdf({query!r}) -> distances {[round(float(s), 3) for _, s in results]}")
        for doc, score in results:
            found_chunks.append({
                "query": query,
                "text": doc.page_content,
                "page": doc.metadata.get("page", 0) + 1,
                "distance": round(float(score), 4),
            })
        return "\n\n".join(doc.page_content for doc, _ in results)

    agent = create_agent(
        model=get_llm(provider),
        tools=[search_pdf],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=memory,
    )
    return agent, found_chunks


def ask(agent, question: str, session_id: str) -> str:
    """Send one message. The same session_id = the agent remembers earlier messages."""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": session_id}},
    )
    return result["messages"][-1].text


# Quick test from the terminal:  python core_rag.py
if __name__ == "__main__":
    db = build_vector_db("../data/sample_syllabus.pdf")
    agent, chunks = build_agent("groq", db)
    print("ANSWER:", ask(agent, "How are the internal marks distributed?", "test"))
    print("ANSWER:", ask(agent, "And how many of those marks are for attendance?", "test"))
