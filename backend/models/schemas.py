from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class EvidenceChunk(BaseModel):
    chunk_id: str
    document_name: str
    source: str
    source_type: str = "General"
    page_number: Optional[int] = 1
    text: str
    relevance_score: float = 0.0
    stance: str = "NEUTRAL"  # SUPPORTING, CONTRADICTING, NEUTRAL
    credibility_score: float = 0.70

class ClaimVerificationRequest(BaseModel):
    claim: str = Field(..., min_length=3, description="The claim or headline to verify")
    llm_provider: Optional[str] = Field("auto", description="auto, gemini, groq, openai, ollama, offline")
    api_key: Optional[str] = Field(None, description="Optional custom API key for this request")
    confidence_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(5, ge=1, le=20)

class SimilarClaim(BaseModel):
    id: str
    claim: str
    verdict: str
    confidence_score: float
    timestamp: str
    similarity_score: float

class VerificationResponse(BaseModel):
    id: str
    claim: str
    verdict: str  # SUPPORTED, REFUTED, MISLEADING, INSUFFICIENT EVIDENCE
    confidence_score: float  # 0 to 100
    explanation: str
    key_reasoning: str
    supporting_evidence: List[EvidenceChunk] = []
    contradicting_evidence: List[EvidenceChunk] = []
    neutral_evidence: List[EvidenceChunk] = []
    retrieved_sources: List[Dict[str, Any]] = []
    source_credibility_score: float = 0.0  # 0 to 100
    evidence_agreement_score: float = 0.0  # 0 to 100
    agreement_analysis: str = ""
    similar_claim_found: Optional[SimilarClaim] = None
    llm_provider_used: str = "offline-nli"
    timestamp: str

class DocumentItem(BaseModel):
    id: str
    filename: str
    source: str
    source_type: str
    file_type: str
    upload_date: str
    number_of_chunks: int
    processing_status: str

class AnalyticsData(BaseModel):
    total_claims_verified: int
    verdict_counts: Dict[str, int]
    average_confidence: float
    average_source_credibility: float
    total_documents: int
    total_chunks: int
    top_sources: List[Dict[str, Any]]
