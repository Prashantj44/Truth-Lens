# TruthLens — Docker Setup Guide

## Architecture

TruthLens runs as a **single Docker container** that hosts:

- **FastAPI backend** (Python 3.11) — API routes, verification logic, ChromaDB, SQLite
- **Static SPA frontend** — `index.html` served by the same FastAPI process
- **ML models** — `sentence-transformers/all-MiniLM-L6-v2` (pre-downloaded at build time)
- **Cloud LLM integration** — Gemini, Groq, OpenAI (via API keys in `.env`)

```
┌─────────────────────────────────────────┐
│            Docker Container             │
│                                         │
│  uvicorn (port 8000)                    │
│    ├── GET /          → index.html      │
│    ├── GET /api/*     → FastAPI routes  │
│    ├── GET /health    → health check    │
│    └── POST /api/verify → fact-check    │
│                                         │
│  Volumes:                               │
│    /app/chroma_db/        → ChromaDB    │
│    /app/backend/database/ → SQLite      │
│    /app/data/uploads/     → User files  │
└─────────────────────────────────────────┘
```

## Prerequisites

- **Docker Desktop** (Windows/Mac/Linux) with Linux containers enabled
- A Gemini API key (or Groq/OpenAI) — optional but recommended for cloud LLM reasoning

## Quick Start

### 1. Create your `.env` file

```powershell
Copy-Item .env.example .env
# Then edit .env with your API keys
```

### 2. Build and Start

```powershell
docker compose up --build
```

First build takes ~3–5 minutes (downloads Python packages + ML models).
Subsequent builds use Docker layer caching and take seconds.

### 3. Open the Application

```
http://localhost:8000
```

## Commands Reference

| Action | Command |
|--------|---------|
| **Build & start** | `docker compose up --build` |
| **Start (detached)** | `docker compose up -d` |
| **Stop** | `docker compose down` |
| **View logs** | `docker compose logs -f` |
| **Restart** | `docker compose restart` |
| **Rebuild from scratch** | `docker compose build --no-cache` |
| **Check status** | `docker ps` |
| **Health check** | `curl http://localhost:8000/health` |

## Environment Variables

All environment variables are documented in `.env.example`. Key variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Recommended | Google Gemini API key for cloud LLM reasoning |
| `GROQ_API_KEY` | Optional | Groq API key (fallback LLM) |
| `OPENAI_API_KEY` | Optional | OpenAI API key (fallback LLM) |
| `DEFAULT_LLM_PROVIDER` | Optional | `auto` (default), `gemini`, `groq`, `openai`, `offline` |

If no API keys are set, TruthLens uses its built-in offline NLI engine.

## Data Persistence

Three named Docker volumes preserve data across container restarts:

| Volume | Container Path | Contents |
|--------|---------------|----------|
| `truthlens-chroma-data` | `/app/chroma_db` | ChromaDB vector database |
| `truthlens-sqlite-data` | `/app/backend/database` | SQLite history & analytics |
| `truthlens-uploads` | `/app/data/uploads` | User-uploaded documents |

### Verify persistence:

```powershell
# Start, verify a claim, stop, restart
docker compose up -d
# ... use the app, verify claims ...
docker compose down
docker compose up -d
# Check that history and verified claims are still there
```

### Reset all data:

```powershell
docker compose down -v    # -v removes volumes
docker compose up --build
```

## Troubleshooting

### Container won't start

```powershell
docker compose logs -f
```

Check for:
- Missing `.env` file → copy from `.env.example`
- Port 8000 already in use → change `ports: "8001:8000"` in `docker-compose.yml`

### Frontend shows JSON instead of UI

The `index.html` must be at the project root. If it's missing, the API returns JSON.

### "No module named 'chromadb'"

Rebuild the image:
```powershell
docker compose build --no-cache
```

### Model download slow on first request

The embedding model is pre-downloaded during `docker build`. If you see download logs at runtime, rebuild:
```powershell
docker compose build --no-cache
```

### Vercel deployment still works?

Yes. Docker files (`Dockerfile`, `docker-compose.yml`, `.dockerignore`) are ignored by Vercel. The `vercel.json`, `api/index.py`, and `index.html` remain untouched.

## Vercel Compatibility

Docker and Vercel coexist in the same repository:

| Deployment | Entry Point | Config |
|-----------|-------------|--------|
| **Vercel** | `api/index.py` → `backend.main:app` | `vercel.json` |
| **Docker** | `uvicorn backend.main:app` | `Dockerfile` + `docker-compose.yml` |

The `IS_VERCEL` flag in `backend/config.py` automatically adjusts paths:
- **Vercel**: uses `/tmp/` for ChromaDB and SQLite (ephemeral)
- **Docker/Local**: uses project-relative paths with persistent volumes
