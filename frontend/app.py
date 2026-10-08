"""
app.py — a simple chat UI for our PDF RAG app.

Run:  streamlit run app.py      (start the FastAPI backend first!)
"""

import os
import uuid

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Chat with your PDF", page_icon="📄")
st.title("📄 Chat with your PDF")
st.caption("RAG agent: FAISS + LangChain + LangGraph + Gemini / Groq")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())  # the agent's memory is stored under this id

# ---------------- Sidebar: upload + model choice ----------------
with st.sidebar:
    st.header("1. Upload a PDF")
    pdf = st.file_uploader("Choose a PDF file", type="pdf")
    if pdf and st.button("Process PDF"):
        with st.spinner("Reading → splitting → turning text into vectors..."):
            try:
                r = requests.post(f"{BACKEND_URL}/upload", files={"file": (pdf.name, pdf.getvalue(), "application/pdf")})
                if r.ok:
                    info = r.json()
                    st.success(f"Done! {info['chunks']} chunks, each stored as a vector of {info['vector_size']} numbers.")
                else:
                    st.error(r.json()["detail"])
            except requests.ConnectionError:
                st.error("Backend is not running. Start it with: uvicorn main:app --port 8000")

    st.header("2. Pick an LLM")
    provider = st.radio("Provider", ["groq", "gemini"])

    if st.button("🧹 New chat (clear memory)"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()


# ---------------- Chat ----------------
def show_trace(sources):
    """The 'X-ray' panel: what did the agent search for, and what did FAISS return?"""
    with st.expander("🔬 Core System Technical Ingress Trace"):
        if not sources:
            st.write("The agent answered **without** searching the PDF (e.g. from memory).")
        for i, src in enumerate(sources, start=1):
            st.markdown(f"**Chunk {i}** · searched for `{src['query']}` · page {src['page']} · "
                        f"distance `{src['distance']}` (lower = more similar)")
            st.text(src["text"])


for msg in st.session_state.messages:  # re-draw the conversation so far
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            show_trace(msg["sources"])

question = st.chat_input("Ask a question about your PDF")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("The agent is thinking..."):
            try:
                payload = {"question": question, "provider": provider, "session_id": st.session_state.session_id}
                r = requests.post(f"{BACKEND_URL}/query", json=payload)
            except requests.ConnectionError:
                r = None
        if r is not None and r.ok:
            result = r.json()
            st.markdown(result["answer"])
            show_trace(result["sources"])
            st.session_state.messages.append({"role": "assistant", "content": result["answer"], "sources": result["sources"]})
        else:
            st.error(r.json()["detail"] if r is not None else "Backend is not running.")
