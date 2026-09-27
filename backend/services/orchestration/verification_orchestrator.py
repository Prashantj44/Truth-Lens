from typing import Dict, Any, Optional
import uuid
from datetime import datetime, timezone
from backend.models.schemas import ClaimVerificationRequest, VerificationResponse, EvidenceChunk
from backend.services.orchestration.claim_parser import claim_parser
from backend.services.orchestration.source_planner import source_planner
from backend.services.orchestration.evidence_retriever import evidence_retriever
from backend.services.orchestration.recommendation_engine import recommendation_engine
from backend.services.orchestration.policy_engine import policy_engine
from backend.services.llm_service import llm_service

class VerificationOrchestrator:
    def __init__(self):
        pass
        
    def process(self, request: ClaimVerificationRequest) -> VerificationResponse:
        claim_str = request.claim
        
        # 1. Parse structured claim
        structured_claim = claim_parser.parse(claim_str)
        
        # 2. Plan Source Strategy
        strategy = source_planner.get_source_strategy(structured_claim)
        
        # 3. Retrieve Evidence
        raw_chunks = evidence_retriever.retrieve(claim_str, strategy)
        
        # Inject client context if provided
        if request.context_text and len(request.context_text.strip()) > 20:
            raw_chunks.insert(0, {
                "chunk_id": f"client-context-{abs(hash(request.context_text)) % 100000}",
                "document_name": "Source Article from Client",
                "source": "Directly Supplied Context from Client",
                "source_type": "Direct Client Context",
                "text": request.context_text,
                "relevance_score": 100.0,
                "credibility_score": 100.0
            })
            
        # 4. Analyze Entailment (using LLM Service to read the chunks against the claim)
        # Note: In a fully rewritten system, LLM Service itself would use structured policy, 
        # but here we use it to extract properties and let PolicyEngine decide.
        if not raw_chunks:
            analysis = {
                "verdict": "INSUFFICIENT_EVIDENCE",
                "explanation": "No verified evidence was found for this claim.",
                "key_reasoning": "Vector index and live search returned 0 relevant chunks.",
                "chunk_classifications": [],
                "entity_match": "Unverified",
                "temporal_match": "Unverified"
            }
        else:
            analysis = llm_service.verify_with_llm(claim_str, raw_chunks)
            
        # 5. Apply Policy Engine Rules
        claim_info = {
            "currentness_required": structured_claim.currentness_required,
            "claim_type": structured_claim.claim_type
        }
        policy_result = policy_engine.determine_verdict(claim_info, analysis)
        
        # 6. Build Evidence Lists
        supporting = []
        contradicting = []
        neutral = []
        
        for cls in analysis.get("chunk_classifications", []):
            matched_c = next((c for c in raw_chunks if str(c.get("chunk_id")) == str(cls.get("chunk_id"))), None)
            if matched_c:
                ec = EvidenceChunk(
                    id=str(uuid.uuid4()),
                    chunk_id=str(matched_c.get("chunk_id", "")),
                    document_name=matched_c.get("document_name", "Unknown Document"),
                    source=matched_c.get("source", "Unknown Source"),
                    source_type=matched_c.get("source_type", "General"),
                    text=matched_c.get("text", ""),
                    text_excerpt=matched_c.get("text", "")[:200] + "...",
                    stance=cls.get("stance", "NEUTRAL"),
                    credibility_score=matched_c.get("credibility_score", 0.70),
                    publication_date=matched_c.get("publication_date"),
                    retrieval_date=matched_c.get("retrieval_date")
                )
                
                if ec.stance == "SUPPORTING":
                    supporting.append(ec)
                elif ec.stance == "CONTRADICTING":
                    contradicting.append(ec)
                else:
                    neutral.append(ec)
                    
        # 7. Generate Recommendation
        recommendation = recommendation_engine.generate(structured_claim, strategy)
        
        # 8. Calculate Verification Strength
        verification_strength = "Insufficient"
        evidence_agreement_score = 0.0
        
        if supporting or contradicting:
            if policy_result["verdict"] == "VERIFIED" and supporting and not contradicting:
                verification_strength = "Strong"
                evidence_agreement_score = 100.0
            elif policy_result["verdict"] == "REFUTED" and contradicting:
                verification_strength = "Strong"
                evidence_agreement_score = 100.0
            elif policy_result["verdict"] == "CONFLICTING EVIDENCE":
                verification_strength = "Moderate"
                evidence_agreement_score = 50.0
            else:
                verification_strength = "Weak"
                evidence_agreement_score = 30.0

        # Construct Final Response
        return VerificationResponse(
            id=str(uuid.uuid4()),
            claim=claim_str,
            claim_type=structured_claim.claim_type,
            verdict=policy_result["verdict"],
            confidence_score=analysis.get("confidence_score", 0.0),
            verification_strength=verification_strength,
            explanation=policy_result["explanation"],
            key_reasoning=policy_result["key_reasoning"],
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            neutral_evidence=neutral,
            source_credibility_score=0.0, # Not randomly assigning 93.9% anymore
            evidence_agreement_score=evidence_agreement_score,
            verification_recommendation=recommendation,
            entity_match=policy_result["entity_match"],
            temporal_match=policy_result["temporal_match"],
            country_match=policy_result["country_match"],
            role_match=policy_result["role_match"],
            llm_provider_used=analysis.get("llm_provider_used", "Orchestrator"),
            timestamp=datetime.now(timezone.utc).isoformat()
        )

verification_orchestrator = VerificationOrchestrator()
