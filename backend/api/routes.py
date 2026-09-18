import os
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from backend.models.schemas import (
    ClaimVerificationRequest,
    VerificationResponse,
    DocumentItem,
    AnalyticsData
)
from backend.services.verification_service import verification_service
from backend.services.document_processor import process_document, detect_source_type
from backend.services.retrieval_service import retrieval_service
from backend.database.db import (
    save_document,
    get_all_documents,
    get_document_by_id,
    delete_document_by_id,
    get_verification_history,
    get_verification_by_id,
    get_analytics
)
from backend.config import UPLOADS_DIR, TRUSTED_DOCS_DIR
from backend.services.news_sync_service import news_sync_service

router = APIRouter(prefix="/api")

@router.post("/verify", response_model=VerificationResponse)
def verify_claim(request: ClaimVerificationRequest):
    """
    Core fact-verification endpoint.
    Retrieves evidence from ChromaDB, reranks chunks, and provides explainable verdict.
    """
    try:
        response = verification_service.verify(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    source_name: Optional[str] = Form(None),
    source_type: Optional[str] = Form(None)
):
    """
    Accepts PDF, TXT, MD documents, extracts text, generates chunks and vector embeddings.
    """
    filename = file.filename
    ext = Path(filename).suffix.lower()
    if ext not in [".pdf", ".txt", ".md", ".json"]:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload PDF, TXT, or MD.")

    # Save file to uploads directory
    target_path = UPLOADS_DIR / filename
    try:
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Process and chunk document
    inferred_source = source_name or filename
    inferred_type = source_type or detect_source_type(filename, inferred_source)
    chunks = process_document(target_path, source_name=inferred_source, source_type=inferred_type)

    if not chunks:
        raise HTTPException(status_code=400, detail="No readable text could be extracted from the document.")

    # Index into ChromaDB vector database
    indexed_count = retrieval_service.add_chunks(chunks)

    # Save to SQLite document registry
    doc_id = save_document(
        filename=filename,
        source=inferred_source,
        source_type=inferred_type,
        file_type=ext.replace(".", "").upper(),
        number_of_chunks=indexed_count
    )

    return {
        "status": "success",
        "document_id": doc_id,
        "filename": filename,
        "source": inferred_source,
        "source_type": inferred_type,
        "chunks_indexed": indexed_count
    }

@router.get("/documents", response_model=List[DocumentItem])
def list_documents():
    """Returns all documents registered in the knowledge base."""
    docs = get_all_documents()
    return [
        DocumentItem(
            id=d["id"],
            filename=d["filename"],
            source=d["source"],
            source_type=d["source_type"],
            file_type=d["file_type"],
            upload_date=d["upload_date"],
            number_of_chunks=d["number_of_chunks"],
            processing_status=d["processing_status"]
        ) for d in docs
    ]

@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    """Deletes a document and its corresponding vector chunks."""
    doc = get_document_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Remove chunks from vector store
    retrieval_service.delete_document_chunks(doc["filename"])

    # Remove physical upload if present
    file_path = UPLOADS_DIR / doc["filename"]
    if file_path.exists():
        try:
            file_path.unlink()
        except Exception:
            pass

    # Delete from DB
    delete_document_by_id(doc_id)
    return {"status": "success", "message": f"Document {doc['filename']} deleted successfully."}

@router.get("/history")
def list_history(limit: int = 50):
    """Retrieves fact verification history."""
    return get_verification_history(limit=limit)

@router.get("/history/{v_id}")
def get_history_detail(v_id: str):
    """Retrieves full details of a specific past verification."""
    item = get_verification_by_id(v_id)
    if not item:
        raise HTTPException(status_code=404, detail="Verification record not found.")
    return item

@router.get("/analytics", response_model=AnalyticsData)
def analytics_dashboard():
    """Provides statistical aggregations and breakdown of fact checks."""
    data = get_analytics()
    return AnalyticsData(**data)

@router.post("/seed")
def seed_trusted_knowledge_base():
    """
    Ingests and indexes the preloaded trusted fact dossiers
    (Economy/GDP, Climate/Renewables, Medical/Vaccines, AI Benchmarks).
    """
    seeded_docs = []
    total_chunks = 0

    if not TRUSTED_DOCS_DIR.exists():
        raise HTTPException(status_code=404, detail="Trusted docs directory not found.")

    for doc_file in TRUSTED_DOCS_DIR.glob("*.txt"):
        filename = doc_file.name
        source_type = detect_source_type(filename)
        chunks = process_document(doc_file, source_name=filename.replace(".txt", "").replace("_", " ").title(), source_type=source_type)
        if chunks:
            # First remove any prior chunks with this filename to prevent duplicate index entries
            retrieval_service.delete_document_chunks(filename)
            indexed_cnt = retrieval_service.add_chunks(chunks)
            doc_id = save_document(
                filename=filename,
                source=filename.replace(".txt", "").replace("_", " ").title(),
                source_type=source_type,
                file_type="TXT",
                number_of_chunks=indexed_cnt
            )
            seeded_docs.append({
                "id": doc_id,
                "filename": filename,
                "chunks": indexed_cnt
            })
            total_chunks += indexed_cnt

    return {
        "status": "success",
        "message": f"Successfully indexed {len(seeded_docs)} curated dossiers ({total_chunks} total evidence chunks).",
        "seeded_documents": seeded_docs,
        "total_chunks_indexed": total_chunks
    }

@router.get("/news/status")
def get_news_status():
    """
    Returns the real-time health and synchronization stats of the daily news engine,
    including total ingested articles, active RSS feeds, and recent headlines.
    """
    return news_sync_service.get_status()

@router.post("/news/sync")
def sync_daily_news(max_per_feed: int = 10):
    """
    Manually triggers an immediate pull and indexing of the latest 24h news 
    from trusted sources (Reuters, BBC World, AP, PolitiFact, Google News).
    """
    result = news_sync_service.sync_live_news(max_per_feed=max_per_feed)
    return result

@router.post("/news/train-dataset")
def train_on_existing_news_dataset(max_articles: int = 100):
    """
    Ingests and vector-indexes historical news articles and PolitiFact ground-truth
    records from existing datasets (ISOT Reuters & LIAR).
    """
    result = news_sync_service.train_on_existing_news_dataset(max_articles=max_articles)
    return result

@router.get("/news/trending")
def get_daily_trending_claims():
    """
    Returns today's active news stories and viral claims for 1-click verification.
    """
    return news_sync_service.get_trending_daily_claims()

@router.post("/news/extract-url")
def extract_claim_from_news_url(payload: dict):
    """
    Extracts the main headline and core claim from an online news link.
    """
    url = payload.get("url", "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="Please provide a valid URL.")
    result = news_sync_service.extract_claim_from_url(url)
    return result

@router.post("/news/clean-forward")
def clean_social_forward_message(payload: dict):
    """
    Cleans viral WhatsApp forward boilerplate and extracts the verifiable factual claim.
    """
    text = payload.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Please provide message text.")
    cleaned = news_sync_service.clean_viral_message(text)
    return {"cleaned_claim": cleaned, "original_length": len(text), "cleaned_length": len(cleaned)}


