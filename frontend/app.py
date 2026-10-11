"""
app.py — the chat website for our PDF RAG agent.

Run:  streamlit run app.py      (start the FastAPI backend first!)
The colours and fonts live in .streamlit/config.toml
"""

import os
import time
import uuid

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
MODELS = {"groq": "⚡ Groq · GPT-OSS 20B", "gemini": "✨ Google · Gemini 2.5 Flash"}
STARTER_QUESTIONS = [
    "What is this document about?",
    "Summarize the key points",
    "List the important numbers or specifications",
    "What are the main features or highlights?",
]

st.set_page_config(page_title="DocChat · Chat with any PDF", page_icon="📄", layout="centered")

# ---------------- Memory of this browser tab ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())  # the agent's memory is stored under this id


def get_status():
    """Ask the backend which document is loaded. Returns None if the backend is not running."""
    try:
        return requests.get(f"{BACKEND_URL}/status", timeout=3).json()
    except requests.RequestException:
        return None


def new_chat():
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())


status = get_status()
has_document = bool(status and status["document"])

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("📄 DocChat")
    st.caption("Ask questions about **any** PDF: a syllabus, a bike brochure, a manual, a report…")
    if status is None:
        st.badge("Backend offline", icon=":material/cloud_off:", color="red")
    else:
        st.badge("Backend online", icon=":material/check_circle:", color="green")

    st.subheader("1 · Your document")
    if has_document:
        with st.container(border=True):
            st.markdown(f"**📄 {status['document']}**")
            st.markdown(f":blue-badge[{status['chunks']} chunks] :violet-badge[{status['vector_size']}-number vectors]")

    pdf = st.file_uploader("Upload a PDF", type="pdf", label_visibility="collapsed")
    if pdf and st.button("Process PDF", type="primary", icon=":material/bolt:", width="stretch"):
        with st.status("Processing your PDF…", expanded=True) as progress:
            st.write("📖 Reading the pages")
            st.write("✂️ Cutting the text into chunks")
            st.write("🔢 Turning each chunk into numbers (embeddings)")
            try:
                r = requests.post(f"{BACKEND_URL}/upload", files={"file": (pdf.name, pdf.getvalue(), "application/pdf")})
                if r.ok:
                    progress.update(label="Ready! Ask me anything 🎉", state="complete", expanded=False)
                    new_chat()                     # new document = new conversation
                    st.rerun()
                else:
                    progress.update(label=r.json()["detail"], state="error")
            except requests.ConnectionError:
                progress.update(label="Backend is not running", state="error")

    st.subheader("2 · AI model")
    provider = st.segmented_control(
        "LLM provider", options=list(MODELS), format_func=lambda p: MODELS[p].split(" · ")[0],
        default="groq", required=True, label_visibility="collapsed",
    )
    st.caption(MODELS[provider])

    st.space("small")
    st.button("New chat", icon=":material/add_comment:", on_click=new_chat, width="stretch")
    st.caption("Built with LangChain · LangGraph · FAISS · Hugging Face 🤗")


# ---------------- Header ----------------
st.title("Chat with your PDF")
st.markdown(":violet-badge[📄 PDF] → :blue-badge[✂️ Chunks] → :green-badge[🔢 Embeddings] → "
            ":orange-badge[🔍 FAISS] → :red-badge[🤖 Agent + Memory]")


def show_sources(sources):
    """The 'X-ray' panel: what the agent searched for, and which chunks FAISS returned."""
    if not sources:
        st.caption("💭 Answered from the conversation, without searching the document.")
        return
    pages = sorted({s["page"] for s in sources})
    st.caption("📑 Sources: " + ", ".join(f"page {p}" for p in pages))
    with st.expander("🔬 Core System Technical Ingress Trace"):
        for i, src in enumerate(sources, start=1):
            match = max(0.0, min(1.0, 1 - src["distance"] / 2))  # for normalized vectors: cosine = 1 - distance/2
            with st.container(border=True):
                st.markdown(f"**Chunk {i}** · page {src['page']} · searched for *“{src['query']}”*")
                st.progress(match, text=f"Match {match:.0%}  ·  distance {src['distance']} (lower = closer)")
                st.code(src["text"], language=None, wrap_lines=True)


def typewriter(text):
    """Show the answer word by word (just for the effect: the full answer has already arrived)."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.005)


# ---------------- Welcome screen (before the first question) ----------------
question = None
if status is None:
    st.error("The backend is not running. Start it with:  `cd backend && uvicorn app.main:app --port 8000`", icon="🔌")
elif not st.session_state.messages:
    with st.container(border=True):
        if not has_document:
            st.markdown("#### 👋 Welcome!")
            st.markdown("1. **Upload** any PDF in the sidebar and click **Process PDF**\n"
                        "2. **Pick** an AI model\n"
                        "3. **Ask** anything. The agent searches your PDF and remembers the conversation")
        else:
            st.markdown(f"#### 💡 Try asking about *{status['document']}*")
            with st.container(horizontal=True):
                for suggestion in STARTER_QUESTIONS:
                    if st.button(suggestion, icon=":material/chat_bubble:"):
                        question = suggestion

# ---------------- Chat history ----------------
AVATARS = {"user": "🧑‍🎓", "assistant": "🤖"}
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        st.markdown(msg["content"])
        if "sources" in msg:
            show_sources(msg["sources"])

# ---------------- New question ----------------
typed = st.chat_input("Ask a question about your PDF…" if has_document else "Upload a PDF first ⬅️",
                      disabled=not has_document)
question = typed or question

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(question)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        try:
            with st.spinner("Searching the document and thinking…"):
                payload = {"question": question, "provider": provider, "session_id": st.session_state.session_id}
                r = requests.post(f"{BACKEND_URL}/query", json=payload)
            if r.ok:
                result = r.json()
                st.write_stream(typewriter(result["answer"]))
                show_sources(result["sources"])
                st.session_state.messages.append(
                    {"role": "assistant", "content": result["answer"], "sources": result["sources"]})
            else:
                st.error(r.json()["detail"], icon="⚠️")
        except requests.ConnectionError:
            st.error("Lost connection to the backend.", icon="🔌")
