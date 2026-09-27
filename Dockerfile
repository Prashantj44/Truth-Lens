# ============================================================
# TruthLens — Dockerfile
# Single-container deployment: FastAPI + static SPA + ChromaDB
# ============================================================
# Architecture:
#   - Python 3.11 slim base (keeps image ~500MB with ML models)
#   - Installs sentence-transformers, chromadb, and cloud LLM SDKs
#   - Pre-downloads the embedding model at build time for fast startup
#   - Serves both API (/api/*) and frontend (index.html) from uvicorn
#   - ChromaDB and SQLite persist via Docker volumes
# ============================================================

FROM python:3.11-slim AS base

# Prevent Python from buffering stdout/stderr (immediate log output)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies required by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- Dependencies ----------
# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies + runtime extras not in requirements.txt
# (chromadb and sentence-transformers are lazy-imported but needed for full functionality)
RUN pip install --no-cache-dir -r requirements.txt \
    chromadb>=0.4.0 \
    sentence-transformers>=2.2.0

# ---------- Pre-download ML models ----------
# Downloads the embedding model at build time so the container starts instantly.
# This avoids a ~100MB download on every first request.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# ---------- Application code ----------
COPY . .

# Create directories that the app expects
RUN mkdir -p /app/data/uploads \
             /app/data/trusted_docs \
             /app/backend/database \
             /app/chroma_db

# ---------- Runtime ----------
# Default port — matches backend/main.py
EXPOSE 8000

# Health check: polls the /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; r=requests.get('http://localhost:8000/health'); exit(0 if r.status_code==200 else 1)" || exit 1

# Launch uvicorn (non-reload for production stability)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
