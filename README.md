# 📄🤖 Chat with your PDF: an introduction to Generative AI

A hands-on workshop for engineering students. You will learn how an AI can answer questions about **your own documents**, using a technique called **RAG (Retrieval-Augmented Generation)**, and you will build a chatbot **agent with memory**.

```
PDF ──► chunks ──► embeddings (numbers) ──► FAISS ──► 🔍 search
                                                        │
             you ──► 🤖 Agent (LLM + system prompt + memory + search tool) ──► answer
```

## What's inside

```
pdf_rag/
├── notebooks/
│   ├── 01_rag_pipeline_exploration.ipynb   ← PDF → chunks → embeddings → pictures → search
│   ├── 02_agentic_workflow.ipynb           ← LLM → system prompt → memory → agent
│   └── 03_rag_agent_end_to_end.ipynb       ← all together: LLM alone vs Classic RAG vs Agentic RAG
├── backend/
│   ├── requirements.txt
│   ├── Dockerfile
│   └── app/                ← the Python package (one file = one job)
│       ├── config.py       ← all settings and API keys in one place
│       ├── prompts.py      ← the system prompt (the agent's instructions)
│       ├── vectordb.py     ← PDF → chunks → embeddings → FAISS
│       ├── llm.py          ← loads Groq or Gemini
│       ├── tools.py        ← functions the agent may call (search_pdf)
│       ├── agent.py        ← LLM + prompt + tools + memory = the agent
│       ├── schemas.py      ← shape of the API's JSON requests/responses
│       └── main.py         ← FastAPI routes (/upload, /query, /status)
├── frontend/       ← Streamlit chat app
├── data/           ← sample_syllabus.pdf (an example document, not an official KTU document)
├── environment.yml ← conda environment
├── .env.example    ← template for your API keys
└── docker-compose.yml
```

---

## Step 1: Get your free API keys 🔑

### Google Gemini
1. Go to **Google AI Studio**: <https://aistudio.google.com>
2. Click **Get API key** → **Create API key**
3. Choose (or create) a Google Cloud project
4. Copy the key (it starts with `AIza...`)

### Groq
1. Go to the **Groq Console**: <https://console.groq.com>
2. Open **API Keys** → **Create API Key**, and name it `KTU_Workshop`
3. Copy the key (it starts with `gsk_...`). It is shown only once!

### Save them in a `.env` file
```bash
cp .env.example .env
```
Open `.env` and paste your keys:
```
GEMINI_API_KEY=AIza...
GROQ_API_KEY=gsk_...
```
> ⚠️ Never share your `.env` file or upload it to GitHub.

---

## Step 2: Set up Python 🐍

### With conda (recommended)
```bash
conda env create -f environment.yml
conda activate rag-workshop
python -m ipykernel install --user --name rag-workshop --display-name "Python (rag-workshop)"
```

### Or with pip
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt jupyterlab matplotlib scikit-learn
```

> Each notebook also starts with a `%pip install` cell, so you can install from inside Jupyter too.
>
> The embedding model comes from **Hugging Face** and needs PyTorch. `requirements.txt` automatically installs the small **CPU-only** PyTorch (~200 MB) instead of the 2–3 GB GPU version.

---

## Step 3: Run the notebooks 📓
```bash
jupyter lab
```
Open the notebooks in order and choose the **Python (rag-workshop)** kernel.

| Notebook | You will learn |
|---|---|
| **01**: PDF to vectors | installation · libraries · document loading · chunking · embedding model · embeddings · visualization (a "map of meaning") · FAISS search |
| **02**: the agent | LLM model · model loading · system prompt · memory · tools · a ReAct agent with memory (LangChain + LangGraph) |
| **03**: all together 🏆 | a 3-round competition on the same questions (🧠 LLM alone vs 📚 Classic RAG vs 🤖 Agentic RAG with memory), ending with a scoreboard |

---

## Step 4: Run the chat app 💬
Open **two terminals** (both with `conda activate rag-workshop`):

```bash
# Terminal 1: the backend API
cd backend
uvicorn app.main:app --reload --port 8000
```
```bash
# Terminal 2: the chat website
cd frontend
streamlit run app.py
```
Open <http://localhost:8501>, upload a PDF, and start chatting. Use the starter questions, or type your own.
Click **🔬 Core System Technical Ingress Trace** under an answer to see which PDF chunks the agent found, and how close they were.

🎨 **Make it yours:** colours and fonts live in `frontend/.streamlit/config.toml`. Change `primaryColor` and refresh the page!

API docs (try the API from your browser): <http://localhost:8000/docs>

---

## Step 5 (optional): Run everything with Docker 🐳
```bash
docker compose up --build
```
Chat app: <http://localhost:8501> · API: <http://localhost:8000/docs>
Stop it with `Ctrl+C`, then run `docker compose down`.

---

## Bonus: which vector database should I use? 🗄️

| | FAISS (we use this) | ChromaDB | Pinecone / Qdrant |
|---|---|---|---|
| Runs | inside your Python program (in memory) | on your computer, saved to a folder | as a separate cloud service |
| Setup | just `pip install` | just `pip install` | account / server needed |
| Good for | learning, small projects | small apps that should remember data | big real-world products |

---

## Common problems 🛠️

| Problem | Fix |
|---|---|
| `API key` error | check that `.env` exists in the main folder and the key is correct |
| Model "decommissioned" / "not found" | models get retired. Change `GROQ_MODEL` or `GEMINI_MODEL` in `.env` (see <https://console.groq.com/docs/models>) |
| `429` / rate limit | free tier limit reached. Wait a minute or switch provider |
| "Backend is not running" in the app | start Terminal 1 (uvicorn) first |
| Graph picture does not show | drawing the picture needs internet |
| First upload is slow | the Hugging Face embedding model `BAAI/bge-small-en-v1.5` (~130 MB) downloads once |
