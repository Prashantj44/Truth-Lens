import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from backend.models.schemas import (
    ClaimVerificationRequest,
    VerificationResponse,
    EvidenceChunk,
    SimilarClaim
)
from backend.services.retrieval_service import retrieval_service
from backend.services.reranking_service import reranking_service
from backend.services.llm_service import llm_service
from backend.services.live_search_service import live_search_service
from backend.database.db import (
    save_verification,
    find_most_similar_claim
)
from backend.config import RETRIEVAL_TOP_K, RERANK_TOP_K

class VerificationService:
    def verify(self, request: ClaimVerificationRequest) -> VerificationResponse:
        claim = request.claim.strip()
        v_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Step 1: Check for similar previously verified claims
        similar_match = find_most_similar_claim(claim)
        similar_claim_obj = None
        if similar_match:
            similar_claim_obj = SimilarClaim(
                id=similar_match["id"],
                claim=similar_match["claim"],
                verdict=similar_match["verdict"],
                confidence_score=similar_match["confidence_score"],
                timestamp=similar_match["timestamp"],
                similarity_score=similar_match["similarity_score"]
            )

        # Step 2: Check if knowledge base has documents and auto-seed if needed
        total_chunks_available = retrieval_service.count()
        if total_chunks_available == 0:
            try:
                from backend.api.routes import seed_trusted_knowledge_base
                seed_trusted_knowledge_base()
            except Exception as e:
                print(f"[VerificationService] Auto-seed notice: {e}")

        # Delegate entire RAG and NLP logic to the new Agent Orchestrator
        from backend.services.agents.agent_orchestrator import agent_orchestrator
        orchestrator_result = agent_orchestrator.process_claim(claim, request.context_text)

        # Map back to models
        supporting = [EvidenceChunk(**c) for c in orchestrator_result.get("supporting_chunks", [])]
        contradicting = [EvidenceChunk(**c) for c in orchestrator_result.get("contradicting_chunks", [])]
        neutral = [EvidenceChunk(**c) for c in orchestrator_result.get("neutral_chunks", [])]
        
        top_chunks = orchestrator_result.get("top_chunks", [])
        
        # Build retrieved sources
        sources_dict: Dict[str, Dict[str, Any]] = {}
        cred_sum = 0.0
        for c in top_chunks:
            s_name = c.get("source", "Unknown")
            p_num = c.get("page_number", 1)
            cred_val = float(c.get("credibility_score", 0.70))
            cred_sum += cred_val
            
            if s_name not in sources_dict:
                sources_dict[s_name] = {
                    "document_name": c.get("document_name", s_name),
                    "source": s_name,
                    "source_type": c.get("source_type", "General"),
                    "credibility_score": round(cred_val * 100, 1),
                    "pages": [p_num]
                }
            elif p_num not in sources_dict[s_name]["pages"]:
                sources_dict[s_name]["pages"].append(p_num)

        retrieved_sources_list = list(sources_dict.values())
        for s in retrieved_sources_list:
            s["pages"].sort()
            
        avg_credibility = round((cred_sum / max(1, len(top_chunks))) * 100, 1)
        
        total_eval = len(supporting) + len(contradicting)
        if total_eval == 0:
            agreement_score = 50.0
            agreement_analysis = "Neutral context without direct consensus."
        else:
            dominant = max(len(supporting), len(contradicting))
            agreement_score = round((dominant / total_eval) * 100, 1)
            if len(supporting) > 0 and len(contradicting) > 0:
                agreement_analysis = "Source disagreement detected."
            elif len(supporting) > 0:
                agreement_analysis = "High source consensus corroborates the claim."
            else:
                agreement_analysis = "High source consensus refutes the claim."

        response_dict = {
            "id": v_id,
            "claim": claim,
            "verdict": orchestrator_result.get("verdict", "INSUFFICIENT EVIDENCE"),
            "confidence_score": orchestrator_result.get("confidence_score", 50.0),
            "explanation": orchestrator_result.get("explanation", ""),
            "key_reasoning": orchestrator_result.get("key_reasoning", ""),
            "supporting_evidence": [s.model_dump() for s in supporting],
            "contradicting_evidence": [c.model_dump() for c in contradicting],
            "neutral_evidence": [n.model_dump() for n in neutral],
            "retrieved_sources": retrieved_sources_list,
            "source_credibility_score": avg_credibility,
            "evidence_agreement_score": agreement_score,
            "agreement_analysis": agreement_analysis,
            "similar_claim_found": similar_claim_obj.model_dump() if similar_claim_obj else None,
            "llm_provider_used": orchestrator_result.get("llm_provider_used", "LMTA"),
            "timestamp": timestamp
        }

        save_verification(response_dict)

        return VerificationResponse(
            id=v_id,
            claim=claim,
            verdict=response_dict["verdict"],
            confidence_score=response_dict["confidence_score"],
            explanation=response_dict["explanation"],
            key_reasoning=response_dict["key_reasoning"],
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            neutral_evidence=neutral,
            retrieved_sources=retrieved_sources_list,
            source_credibility_score=avg_credibility,
            evidence_agreement_score=agreement_score,
            agreement_analysis=agreement_analysis,
            similar_claim_found=similar_claim_obj,
            llm_provider_used=response_dict["llm_provider_used"],
            timestamp=timestamp
        )

verification_service = VerificationService()
