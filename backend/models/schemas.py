from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class EvidenceChunk(BaseModel):
    id: str = Field(default_factory=lambda: "")
    chunk_id: str = Field(default_factory=lambda: "")
    document_name: str = ""
    source: str = ""
    source_type: str = "General"
    url: Optional[str] = None
    page_number: Optional[int] = 1
    text: str = ""
    text_excerpt: str = ""
    relevance_score: float = 0.0
    stance: str = "NEUTRAL"
    credibility_score: float = 0.70
    publication_date: Optional[str] = None
    retrieval_date: Optional[str] = None
    updated_at: Optional[str] = None
    authority: str = "Unverified"
    primary_or_secondary: str = "Secondary"
    publisher: Optional[str] = None
    entity_match: Optional[str] = None
    temporal_match: Optional[str] = None
    claim_relation: Optional[str] = None
    support_type: Optional[str] = None

class StructuredClaim(BaseModel):
    original_claim: str
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object_entity: Optional[str] = None
    entities: List[str] = []
    country: Optional[str] = None
    organization: Optional[str] = None
    role: Optional[str] = None
    location: Optional[str] = None
    time_expression: Optional[str] = None
    claim_type: str = "AMBIGUOUS"
    currentness_required: bool = False

class VerificationRecommendation(BaseModel):
    source_type: str
    target_domain: str
    queries: List[str] = []
    what_to_check: str
    what_confirms: str
    what_refutes: str

class ClaimVerificationRequest(BaseModel):
    claim: str = Field(..., min_length=3, description="The claim or headline to verify")
    context_text: Optional[str] = Field(None, description="Direct article context provided by a client app (like Echo News) to improve accuracy")
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
    claim_type: str = "AMBIGUOUS"
    verdict: str  # VERIFIED, REFUTED, CONFLICTING EVIDENCE, INSUFFICIENT EVIDENCE, HISTORICAL SUPPORT ONLY, CURRENT STATUS UNVERIFIED, AMBIGUOUS
    confidence_score: float  # 0 to 100 (if retained for compatibility)
    verification_strength: str = "Insufficient" # Strong, Moderate, Weak, Insufficient
    explanation: str
    reason: str = ""
    key_reasoning: str
    entity_analysis: Dict[str, Any] = {}
    temporal_analysis: Dict[str, Any] = {}
    supporting_evidence: List[EvidenceChunk] = []
    contradicting_evidence: List[EvidenceChunk] = []
    neutral_evidence: List[EvidenceChunk] = []
    retrieved_sources: List[Dict[str, Any]] = []
    source_credibility_score: float = 0.0 
    evidence_agreement_score: float = 0.0 
    agreement_analysis: str = ""
    similar_claim_found: Optional[SimilarClaim] = None
    verification_recommendation: Optional[VerificationRecommendation] = None
    retrieved_at: Optional[str] = None
    limitations: List[str] = []
    entity_match: Optional[str] = None
    temporal_match: Optional[str] = None
    country_match: Optional[str] = None
    role_match: Optional[str] = None
    llm_provider_used: str = "orchestrator"
    timestamp: str

class DocumentItem(BaseModel):
    id: str
    filename: str
    source: str
    title: Optional[str] = None
    source_type: str
    file_type: str
    upload_date: str
    created_at: Optional[str] = None
    number_of_chunks: int
    chunks_count: Optional[int] = None
    processing_status: str

    def __init__(self, **data):
        if "title" not in data or not data["title"]:
            data["title"] = data.get("source") or data.get("filename", "Untitled")
        if "chunks_count" not in data or data["chunks_count"] is None:
            data["chunks_count"] = data.get("number_of_chunks", 0)
        if "created_at" not in data or not data["created_at"]:
            data["created_at"] = data.get("upload_date", "")
        super().__init__(**data)

class AnalyticsData(BaseModel):
    total_claims_verified: int
    verdict_counts: Dict[str, int]
    average_confidence: float
    average_source_credibility: float
    total_documents: int
    total_chunks: int
    top_sources: List[Dict[str, Any]]
