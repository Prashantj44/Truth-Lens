import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from backend.config import DB_PATH

def get_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Documents table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            source TEXT NOT NULL,
            source_type TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_hash TEXT,
            upload_date TEXT NOT NULL,
            number_of_chunks INTEGER DEFAULT 0,
            processing_status TEXT DEFAULT 'COMPLETED'
        )
    """)
    
    # Verification history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS verification_history (
            id TEXT PRIMARY KEY,
            claim TEXT NOT NULL,
            verdict TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            explanation TEXT NOT NULL,
            key_reasoning TEXT,
            source_credibility_score REAL DEFAULT 0.0,
            evidence_agreement_score REAL DEFAULT 0.0,
            agreement_analysis TEXT,
            raw_response TEXT,
            timestamp TEXT NOT NULL
        )
    """)
    
    conn.commit()
    conn.close()

# Initialize tables immediately
init_db()

def save_document(
    filename: str,
    source: str,
    source_type: str,
    file_type: str,
    number_of_chunks: int,
    file_hash: str = "",
    doc_id: Optional[str] = None
) -> str:
    doc_id = doc_id or str(uuid.uuid4())
    upload_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO documents 
        (id, filename, source, source_type, file_type, file_hash, upload_date, number_of_chunks, processing_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'COMPLETED')
    """, (doc_id, filename, source, source_type, file_type, file_hash, upload_date, number_of_chunks))
    conn.commit()
    conn.close()
    return doc_id

def get_all_documents() -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents ORDER BY upload_date DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_document_by_id(doc_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def delete_document_by_id(doc_id: str) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

def save_verification(data: Dict[str, Any]) -> str:
    v_id = data.get("id") or str(uuid.uuid4())
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO verification_history 
        (id, claim, verdict, confidence_score, explanation, key_reasoning, source_credibility_score, evidence_agreement_score, agreement_analysis, raw_response, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        v_id,
        data["claim"],
        data["verdict"],
        data["confidence_score"],
        data["explanation"],
        data.get("key_reasoning", ""),
        data.get("source_credibility_score", 0.0),
        data.get("evidence_agreement_score", 0.0),
        data.get("agreement_analysis", ""),
        json.dumps(data),
        data.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()
    return v_id

def get_verification_history(limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, claim, verdict, confidence_score, explanation, timestamp, raw_response
        FROM verification_history 
        ORDER BY timestamp DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    results = []
    for r in rows:
        d = dict(r)
        if d.get("raw_response"):
            try:
                full_obj = json.loads(d["raw_response"])
                results.append(full_obj)
                continue
            except Exception:
                pass
        results.append(d)
    return results

def get_verification_by_id(v_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT raw_response FROM verification_history WHERE id = ?", (v_id,))
    row = cursor.fetchone()
    conn.close()
    if row and row["raw_response"]:
        try:
            return json.loads(row["raw_response"])
        except Exception:
            return None
    return None

def find_most_similar_claim(claim_text: str) -> Optional[Dict[str, Any]]:
    """Simple token and substring overlap check on existing history."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, claim, verdict, confidence_score, timestamp FROM verification_history")
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return None
        
    def clean(s):
        return set(s.lower().replace("?", "").replace(".", "").replace(",", "").split())
        
    c_tokens = clean(claim_text)
    if not c_tokens:
        return None
        
    best_match = None
    best_jaccard = 0.0
    
    for r in rows:
        r_tokens = clean(r["claim"])
        if not r_tokens:
            continue
        intersection = c_tokens.intersection(r_tokens)
        union = c_tokens.union(r_tokens)
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # High similarity match
        if jaccard > best_jaccard and jaccard >= 0.65:
            best_jaccard = jaccard
            best_match = {
                "id": r["id"],
                "claim": r["claim"],
                "verdict": r["verdict"],
                "confidence_score": r["confidence_score"],
                "timestamp": r["timestamp"],
                "similarity_score": round(jaccard * 100, 1)
            }
            
    return best_match

def get_analytics():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Total claims
    cursor.execute("SELECT COUNT(*) as total FROM verification_history")
    total_claims = cursor.fetchone()["total"]
    
    # Verdict counts
    verdicts = {"SUPPORTED": 0, "REFUTED": 0, "MISLEADING": 0, "INSUFFICIENT EVIDENCE": 0}
    cursor.execute("SELECT verdict, COUNT(*) as cnt FROM verification_history GROUP BY verdict")
    for r in cursor.fetchall():
        if r["verdict"] in verdicts:
            verdicts[r["verdict"]] = r["cnt"]
            
    # Average confidence
    cursor.execute("SELECT AVG(confidence_score) as avg_conf, AVG(source_credibility_score) as avg_cred FROM verification_history")
    avg_row = cursor.fetchone()
    avg_conf = round(avg_row["avg_conf"] or 0.0, 1)
    avg_cred = round(avg_row["avg_cred"] or 0.0, 1)
    
    # Total documents & chunks
    cursor.execute("SELECT COUNT(*) as total_docs, SUM(number_of_chunks) as total_chunks FROM documents")
    doc_row = cursor.fetchone()
    total_docs = doc_row["total_docs"] or 0
    total_chunks = doc_row["total_chunks"] or 0
    
    # Top sources
    cursor.execute("SELECT source, source_type, COUNT(*) as doc_count FROM documents GROUP BY source ORDER BY doc_count DESC LIMIT 5")
    top_sources = [dict(r) for r in cursor.fetchall()]
    
    conn.close()
    return {
        "total_claims_verified": total_claims,
        "verdict_counts": verdicts,
        "average_confidence": avg_conf,
        "average_source_credibility": avg_cred,
        "total_documents": total_docs,
        "total_chunks": total_chunks,
        "top_sources": top_sources
    }
