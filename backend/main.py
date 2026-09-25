import asyncio
import os
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

from backend.api.routes import router as api_router, seed_trusted_knowledge_base
from backend.services.retrieval_service import retrieval_service
from backend.services.news_sync_service import news_sync_service
from backend.database.db import init_db
from backend.config import IS_VERCEL

app = FastAPI(
    title="TruthLens: Explainable RAG Fact Verification Engine",
    description="Automated evidence-based fact verification with semantic retrieval, reranking, and explainable AI verdicts.",
    version="2.0.0"
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include API endpoints
app.include_router(api_router)

@app.on_event("startup")
async def startup_event():
    """Initializes SQLite database, auto-seeds dossiers, and launches background news scheduler."""
    try:
        init_db()
    except Exception as e:
        print(f"[TruthLens] DB init notice: {e}")

    try:
        # If vector database is empty, automatically seed initial trusted documents
        current_count = retrieval_service.count()
        if current_count == 0:
            print("[TruthLens] Knowledge base is empty on boot. Seeding initial trusted reference dossiers...")
            seed_trusted_knowledge_base()
            print(f"[TruthLens] Initial seed complete. Chunks indexed: {retrieval_service.count()}")
    except Exception as e:
        print(f"[TruthLens] Notice on startup seeding: {e}")

    # Launch daily news sync background scheduler only in persistent server mode
    if not IS_VERCEL:
        try:
            asyncio.create_task(news_sync_service.start_daily_scheduler())
            print("[TruthLens] Daily live news automated scheduler task launched.")
        except Exception as e:
            print(f"[TruthLens] Failed to launch news scheduler: {e}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "TruthLens Fact Verification Engine",
        "indexed_chunks": retrieval_service.count()
    }

@app.get("/favicon.ico", include_in_schema=False)
async def get_favicon_ico():
    base_dirs = [Path(__file__).resolve().parent.parent / "public", Path(__file__).resolve().parent.parent / "frontend", Path(__file__).resolve().parent.parent]
    for d in base_dirs:
        p = d / "favicon.ico"
        if p.exists():
            return FileResponse(str(p), media_type="image/x-icon")
    return Response(status_code=204)

@app.get("/favicon.svg", include_in_schema=False)
async def get_favicon_svg():
    base_dirs = [Path(__file__).resolve().parent.parent / "public", Path(__file__).resolve().parent.parent / "frontend", Path(__file__).resolve().parent.parent]
    for d in base_dirs:
        p = d / "favicon.svg"
        if p.exists():
            return FileResponse(str(p), media_type="image/svg+xml")
    return Response(status_code=204)

@app.get("/favicon.png", include_in_schema=False)
async def get_favicon_png():
    base_dirs = [Path(__file__).resolve().parent.parent / "public", Path(__file__).resolve().parent.parent / "frontend", Path(__file__).resolve().parent.parent]
    for d in base_dirs:
        p = d / "favicon.png"
        if p.exists():
            return FileResponse(str(p), media_type="image/png")
    return Response(status_code=204)

# Serve Frontend Single-Page Application
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
FRONTEND_DIST = FRONTEND_DIR / "dist"

if FRONTEND_DIST.exists() and (FRONTEND_DIST / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")
    @app.get("/{full_path:path}")
    async def serve_dist(full_path: str):
        file_path = FRONTEND_DIST / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(FRONTEND_DIST / "index.html"))
elif (FRONTEND_DIR / "index.html").exists():
    if (FRONTEND_DIR / "assets").exists():
        app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")

    @app.get("/")
    async def serve_index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        if full_path.startswith("api/") or full_path == "docs" or full_path == "openapi.json" or full_path == "health":
            raise HTTPException(status_code=404, detail=f"API endpoint '/{full_path}' not found")
        file_path = FRONTEND_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(FRONTEND_DIR / "index.html"))
else:
    @app.get("/")
    def index():
        return {
            "name": "TruthLens Fact Verification API",
            "tagline": "See the Truth Behind Every Claim",
            "version": "2.0.0",
            "docs": "/docs",
            "health": "/health"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
