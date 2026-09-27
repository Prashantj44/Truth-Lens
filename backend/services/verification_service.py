import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from backend.models.schemas import (
    ClaimVerificationRequest,
    VerificationResponse,
    SimilarClaim
)
from backend.services.retrieval_service import retrieval_service
from backend.services.orchestration.verification_orchestrator import verification_orchestrator
from backend.database.db import (
    save_verification,
    find_most_similar_claim
)

class VerificationService:
    def verify(self, request: ClaimVerificationRequest) -> VerificationResponse:
        claim = request.claim.strip()

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

        # Step 3: Delegate to the Evidence-First Orchestrator
        orchestrator_response = verification_orchestrator.process(request)
        
        if similar_claim_obj:
             orchestrator_response.similar_claim_found = similar_claim_obj
             
        # Step 4: Extract retrieved sources for UI grouping
        sources_dict: Dict[str, Dict[str, Any]] = {}
        all_chunks = orchestrator_response.supporting_evidence + orchestrator_response.contradicting_evidence + orchestrator_response.neutral_evidence
        
        cred_sum = 0.0
        for c in all_chunks:
            s_name = c.source or "Unknown"
            p_num = c.page_number or 1
            cred_val = float(c.credibility_score or 0.70)
            cred_sum += cred_val
            
            if s_name not in sources_dict:
                sources_dict[s_name] = {
                    "document_name": c.document_name or s_name,
                    "source": s_name,
                    "source_type": c.source_type or "General",
                    "url": c.url or "",
                    "credibility_score": round(cred_val * 100, 1),
                    "pages": [p_num]
                }
            elif p_num not in sources_dict[s_name]["pages"]:
                sources_dict[s_name]["pages"].append(p_num)
            
            # If a source already existed but was missing URL, update it
            if not sources_dict[s_name].get("url") and c.url:
                sources_dict[s_name]["url"] = c.url

        retrieved_sources_list = list(sources_dict.values())
        for s in retrieved_sources_list:
            s["pages"].sort()
            
        orchestrator_response.retrieved_sources = retrieved_sources_list
        orchestrator_response.source_credibility_score = round((cred_sum / max(1, len(all_chunks))) * 100, 1)

        # Step 5: Save verification
        save_verification(orchestrator_response.model_dump())

        return orchestrator_response

verification_service = VerificationService()
