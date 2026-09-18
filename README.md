# 🔍 TruthLens — Explainable Retrieval-Augmented Generation System for Automated Fact Verification

## 📝 Project Summary
TruthLens is an AI-powered fact-verification platform that uses Retrieval-Augmented Generation (RAG) to verify claims against a curated knowledge base of trusted documents. Users enter a claim, and the system retrieves relevant evidence using semantic search, reranks it, analyzes it through an NLI engine (with optional LLM support for Gemini/Groq/OpenAI/Ollama), and provides one of four explainable verdicts: SUPPORTED, REFUTED, MISLEADING, or INSUFFICIENT EVIDENCE.

## ✨ Key Features
- **RAG-based fact verification** with semantic search (sentence-transformers + ChromaDB)
- **Hybrid reranking** (50% semantic + 30% lexical + 20% entity overlap)
- **Multi-provider LLM support** (Google Gemini, Groq, OpenAI, Ollama) with offline NLI fallback
- **Explainable verdicts** with evidence classification and confidence scoring
- **Source credibility assessment**
- **Evidence agreement analysis** (supporting vs contradicting)
- **Similar claim detection** via Jaccard similarity
- **Document upload and management** (PDF/TXT)
- **Verification history tracking** with analytics
- **Beautiful dark glassmorphism React frontend** (CDN-based SPA)

## 🛠️ Tech Stack & Built With
- **Backend:** Python 3.13, FastAPI, Uvicorn
- **Vector DB:** ChromaDB (HNSW, cosine similarity)
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2 (384-dim)
- **NLP:** Offline NLI engine with stance classification
- **LLM (Optional):** Gemini, Groq, OpenAI, Ollama
- **Frontend:** React 18, Tailwind CSS, Chart.js (CDN-based, no npm needed)
- **Database:** SQLite (history, documents, analytics)

## 📁 Project Structure
```text
TruthLens/
├── backend/
│   ├── __init__.py
│   ├── config.py              # Configuration & hyperparameters
│   ├── main.py                # FastAPI app, CORS, auto-seed, frontend serving
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py          # All REST endpoints
│   ├── database/
│   │   ├── __init__.py
│   │   └── db.py              # SQLite operations
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py         # Pydantic request/response models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── document_processor.py   # PDF/TXT extraction, chunking
│   │   ├── embedding_service.py    # SentenceTransformer embeddings
│   │   ├── retrieval_service.py    # ChromaDB vector store operations
│   │   ├── reranking_service.py    # Hybrid reranker
│   │   ├── llm_service.py          # Multi-provider LLM + offline NLI
│   │   └── verification_service.py # End-to-end verification pipeline
│   └── tests/
│       └── test_rag.py        # Integration tests
├── frontend/
│   └── index.html             # Complete React SPA (CDN-based)
├── data/
│   └── trusted_docs/          # Curated fact dossiers
│       ├── economy_gdp_rankings_2025.txt
│       ├── climate_renewable_energy_2024_2025.txt
│       ├── medical_and_public_health_facts.txt
│       └── artificial_intelligence_benchmarks.txt
├── chroma_db/                 # ChromaDB persistent vector store
├── requirements.txt           # Python dependencies
└── README.md
```

## 🏗️ Architecture Diagram
```text
  User Claim
      │
      ▼
  ┌───────────────────┐
  │  Claim Preprocessor│  ← Clean & normalize
  └────────┬──────────┘
           │
           ▼
  ┌───────────────────┐     ┌─────────────────┐
  │  Embedding Service │────▶│  ChromaDB        │
  │  (MiniLM-L6-v2)   │     │  Vector Store    │
  └────────┬──────────┘     │  (26+ chunks)   │
           │                └────────┬────────┘
           ▼                         │
  ┌───────────────────┐              │
  │  Retrieval Service │◀────────────┘
  │  (Top-K retrieval) │
  └────────┬──────────┘
           │
           ▼
  ┌───────────────────┐
  │  Hybrid Reranker   │  ← 50% semantic + 30% lexical + 20% entity
  │  + Credibility     │
  └────────┬──────────┘
           │
           ▼
  ┌───────────────────┐
  │  LLM / NLI Engine  │  ← Verdict determination
  │  (Gemini/Groq/     │
  │   Offline NLI)     │
  └────────┬──────────┘
           │
           ▼
  ┌───────────────────┐
  │  Explainable       │  ← Confidence, agreement, citations
  │  Verdict Response  │
  └───────────────────┘
```

## ⚖️ Verdicts
- ✅ **SUPPORTED** — Evidence confirms the claim (green)
- ❌ **REFUTED** — Evidence contradicts the claim (red)
- ⚠️ **MISLEADING** — Mixed evidence, partially true/false (amber)
- ❓ **INSUFFICIENT EVIDENCE** — Not enough data to verify (indigo)

## 🚀 Quick Start

1. **Clone & enter directory**
2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Start the server:**
   ```bash
   # Windows
   py -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
   
   # Linux/Mac
   python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
   ```
4. **Open browser** at `http://127.0.0.1:8000`
5. The knowledge base auto-seeds on first run (4 curated documents, 26+ chunks).

## 🔌 API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/verify` | Verify a claim against the knowledge base |
| POST | `/api/upload` | Upload a document (PDF/TXT) to knowledge base |
| GET | `/api/documents` | List all indexed documents |
| DELETE | `/api/documents/{id}` | Remove a document |
| GET | `/api/history` | Get verification history |
| GET | `/api/history/{id}` | Get specific verification result |
| GET | `/api/analytics` | Get system analytics |
| POST | `/api/seed` | Re-seed the knowledge base |
| GET | `/health` | Health check |

### Example API Usage:
```bash
curl -X POST http://127.0.0.1:8000/api/verify \
  -H "Content-Type: application/json" \
  -d '{"claim": "India is the third largest economy in the world"}'
```

## 🧠 Optional LLM Configuration
The system works fully offline with the built-in NLI engine. To enhance results with an LLM, set environment variables:
```bash
# Google Gemini (recommended)
set GEMINI_API_KEY=your_key_here

# OR Groq
set GROQ_API_KEY=your_key_here

# OR OpenAI
set OPENAI_API_KEY=your_key_here

# OR Ollama (local)
# Just run ollama serve and the system will detect it
```

## 👥 Team
| Name | Roll No |
|------|--------|
| Riddhi Patil | 52 |
| Prashant Jha | 54 |
| Aditya Soni | 57 |
| Riwan Pereira | 65 |

**Guide:** Dr. Joanne Gomes  
**Institute:** St. Francis Institute of Technology

## 🧪 Tested Verification Results
| Claim | Verdict | Confidence |
|-------|---------|------------|
| India is the 3rd largest economy in the world | MISLEADING | ~75% |
| Renewable energy accounts for over 30% of global electricity | SUPPORTED | ~98% |
| Antibiotics can cure viral infections like the flu | REFUTED | ~93% |
| Flying saucers were discovered in Atlantis | INSUFFICIENT EVIDENCE | ~35% |

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
