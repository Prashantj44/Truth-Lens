import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

IS_VERCEL = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))

BASE_DIR = Path(__file__).resolve().parent.parent
TRUSTED_DOCS_DIR = BASE_DIR / "data" / "trusted_docs"

if IS_VERCEL:
    DATA_DIR = Path("/tmp/data")
    UPLOADS_DIR = DATA_DIR / "uploads"
    CHROMA_PERSIST_DIR = Path("/tmp/chroma_db")
    DB_PATH = Path("/tmp/truthlens.db")
else:
    DATA_DIR = BASE_DIR / "data"
    UPLOADS_DIR = DATA_DIR / "uploads"
    CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"
    DB_PATH = BASE_DIR / "backend" / "database" / "truthlens.db"

# Safely create required directories
for d in [DATA_DIR, TRUSTED_DOCS_DIR, UPLOADS_DIR, CHROMA_PERSIST_DIR, DB_PATH.parent]:
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

# Vector DB & Embeddings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_COLLECTION_NAME = "truthlens_evidence_vault"

# RAG Hyperparameters
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "12"))  # Candidate chunks fetched
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))         # Top chunks passed to LLM
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "450"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "60"))

# Source Credibility Scoring Weights
SOURCE_CREDIBILITY_WEIGHTS = {
    "government": 0.95,
    "official": 0.95,
    "peer-reviewed": 0.95,
    "academic": 0.92,
    "established news": 0.85,
    "reputable media": 0.82,
    "general": 0.65,
    "unknown": 0.50
}

# LLM Providers Configuration
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "auto")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
