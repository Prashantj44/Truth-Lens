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

        # Step 2: Check if knowledge base has documents
        total_chunks_available = retrieval_service.count()
        if total_chunks_available == 0:
            resp_data = {
                "id": v_id,
                "claim": claim,
                "verdict": "INSUFFICIENT EVIDENCE",
                "confidence_score": 10.0,
                "explanation": "No documents are currently indexed in the knowledge base. Please upload documents or seed the knowledge base with trusted reference data to enable verification.",
                "key_reasoning": "Knowledge base vector index is empty (0 indexed chunks).",
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "neutral_evidence": [],
                "retrieved_sources": [],
                "source_credibility_score": 0.0,
                "evidence_agreement_score": 0.0,
                "agreement_analysis": "No evidence available for cross-document agreement calculation.",
                "similar_claim_found": similar_claim_obj,
                "llm_provider_used": "Offline Safeguard",
                "timestamp": timestamp
            }
            save_verification(resp_data)
            return VerificationResponse(**resp_data)

        # Step 3: Semantic Vector Retrieval (Stage 1)
        candidate_chunks = retrieval_service.retrieve(claim, top_k=RETRIEVAL_TOP_K)
        
        if not candidate_chunks:
            resp_data = {
                "id": v_id,
                "claim": claim,
                "verdict": "INSUFFICIENT EVIDENCE",
                "confidence_score": 20.0,
                "explanation": "No relevant evidence chunks could be retrieved for this claim from the existing knowledge base.",
                "key_reasoning": "Semantic search returned zero matches exceeding the minimum similarity baseline.",
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "neutral_evidence": [],
                "retrieved_sources": [],
                "source_credibility_score": 0.0,
                "evidence_agreement_score": 0.0,
                "agreement_analysis": "No documents matched the claim query.",
                "similar_claim_found": similar_claim_obj,
                "llm_provider_used": "Offline Safeguard",
                "timestamp": timestamp
            }
            save_verification(resp_data)
            return VerificationResponse(**resp_data)

        # Step 4: Hybrid Reranking (Stage 2)
        top_chunks = reranking_service.rerank(claim, candidate_chunks, top_k=request.top_k or RERANK_TOP_K)

        # Step 5: Run LLM Verification Reasoning
        llm_result = llm_service.verify_with_llm(
            claim=claim,
            evidence_chunks=top_chunks,
            provider=request.llm_provider or "auto",
            custom_api_key=request.api_key
        )

        # Map stance classifications to chunks
        classifications_map = {
            item["chunk_id"]: item.get("stance", "NEUTRAL")
            for item in llm_result.get("chunk_classifications", [])
        }

        supporting: List[EvidenceChunk] = []
        contradicting: List[EvidenceChunk] = []
        neutral: List[EvidenceChunk] = []

        cred_sum = 0.0
        sources_dict: Dict[str, Dict[str, Any]] = {}

        for c in top_chunks:
            cid = c["chunk_id"]
            stance = classifications_map.get(cid, "NEUTRAL")
            c["stance"] = stance
            cred_val = float(c.get("credibility_score", 0.70))
            cred_sum += cred_val

            # Track unique sources
            s_name = c.get("source", "Unknown")
            p_num = c.get("page_number", 1)
            if s_name not in sources_dict:
                sources_dict[s_name] = {
                    "document_name": c.get("document_name", s_name),
                    "source": s_name,
                    "source_type": c.get("source_type", "General"),
                    "credibility_score": round(cred_val * 100, 1),
                    "pages": [p_num]
                }
            else:
                if p_num not in sources_dict[s_name]["pages"]:
                    sources_dict[s_name]["pages"].append(p_num)

            ev_chunk = EvidenceChunk(
                chunk_id=cid,
                document_name=c.get("document_name", "Unknown"),
                source=s_name,
                source_type=c.get("source_type", "General"),
                page_number=p_num,
                text=c.get("text", ""),
                relevance_score=round(float(c.get("relevance_score", 0.5)) * 100, 1),
                stance=stance,
                credibility_score=round(cred_val * 100, 1)
            )

            if stance == "SUPPORTING":
                supporting.append(ev_chunk)
            elif stance == "CONTRADICTING":
                contradicting.append(ev_chunk)
            else:
                neutral.append(ev_chunk)

        # Compute Aggregate Source Credibility (0 - 100%)
        avg_credibility = round((cred_sum / max(1, len(top_chunks))) * 100, 1)

        # Compute Evidence Agreement Score
        total_eval = len(supporting) + len(contradicting)
        if total_eval == 0:
            agreement_score = 50.0
            agreement_analysis = "Retrieved sources provide neutral background context without direct consensus or dispute."
        else:
            # High agreement if all supporting or all contradicting
            dominant_count = max(len(supporting), len(contradicting))
            agreement_score = round((dominant_count / total_eval) * 100, 1)
            if len(supporting) > 0 and len(contradicting) > 0:
                agreement_analysis = f"Source Disagreement Detected: {len(supporting)} source(s) corroborate aspects of the claim while {len(contradicting)} source(s) register factual contradiction."
            elif len(supporting) > 0:
                agreement_analysis = f"High Source Consensus: All {len(supporting)} evaluable source(s) unanimously corroborate the factual premise."
            else:
                agreement_analysis = f"High Source Consensus: All {len(contradicting)} evaluable source(s) uniformly refute the claim."

        # Insufficient evidence override if confidence is too low or relevance is poor
        final_verdict = llm_result.get("verdict", "INSUFFICIENT EVIDENCE")
        final_confidence = float(llm_result.get("confidence_score", 60.0))

        retrieved_sources_list = list(sources_dict.values())
        for s in retrieved_sources_list:
            s["pages"].sort()

        response_dict = {
            "id": v_id,
            "claim": claim,
            "verdict": final_verdict,
            "confidence_score": round(final_confidence, 1),
            "explanation": llm_result.get("explanation", ""),
            "key_reasoning": llm_result.get("key_reasoning", ""),
            "supporting_evidence": [s.model_dump() for s in supporting],
            "contradicting_evidence": [c.model_dump() for c in contradicting],
            "neutral_evidence": [n.model_dump() for n in neutral],
            "retrieved_sources": retrieved_sources_list,
            "source_credibility_score": avg_credibility,
            "evidence_agreement_score": agreement_score,
            "agreement_analysis": agreement_analysis,
            "similar_claim_found": similar_claim_obj.model_dump() if similar_claim_obj else None,
            "llm_provider_used": llm_result.get("llm_provider_used", "Offline Natural Language Inference Engine"),
            "timestamp": timestamp
        }

        # Step 10: Persist verification to SQLite history
        save_verification(response_dict)

        return VerificationResponse(
            id=v_id,
            claim=claim,
            verdict=final_verdict,
            confidence_score=round(final_confidence, 1),
            explanation=llm_result.get("explanation", ""),
            key_reasoning=llm_result.get("key_reasoning", ""),
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            neutral_evidence=neutral,
            retrieved_sources=retrieved_sources_list,
            source_credibility_score=avg_credibility,
            evidence_agreement_score=agreement_score,
            agreement_analysis=agreement_analysis,
            similar_claim_found=similar_claim_obj,
            llm_provider_used=llm_result.get("llm_provider_used", "Offline Natural Language Inference Engine"),
            timestamp=timestamp
        )

verification_service = VerificationService()
