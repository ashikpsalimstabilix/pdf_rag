FROM python:3.11-slim

# Show print() output in `docker logs` immediately
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the embedding model ONCE while building the image...
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
# ...so the running app never has to contact Hugging Face again
ENV HF_HUB_OFFLINE=1

COPY backend/ .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
