import os
from typing import List, Dict, Any, Optional
from backend.services.retrieval_service import retrieval_service
from backend.services.live_search_service import live_search_service
from backend.services.nlp.nli_evaluator import nli_evaluator
from backend.config import RERANK_TOP_K

class AgentOrchestrator:
    """
    Coordinates the Router, Researcher, and Synthesizer/Judge tasks.
    """
    def __init__(self):
        pass

    def _router_analyze(self, claim: str) -> Dict[str, Any]:
        """
        Router Agent logic: Decides where to search based on the claim.
        For now, we use a heuristic router to save API calls, 
        but this can be swapped with a Groq LLM call.
        """
        claim_lower = claim.lower()
        if "president" in claim_lower or "election" in claim_lower or "minister" in claim_lower:
            domain = "politics"
        elif "cure" in claim_lower or "disease" in claim_lower or "vaccine" in claim_lower:
            domain = "science"
        else:
            domain = "general"

        # Always do a hybrid search (local + live)
        return {
            "domain": domain,
            "search_local": True,
            "search_live": True
        }

    def _researcher_gather(self, claim: str, instructions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Researcher Agent logic: Gathers chunks from vector DB and live web.
        """
        evidence_chunks = []
        
        if instructions.get("search_local"):
            # Fetch from local ChromaDB
            local_chunks = retrieval_service.retrieve(claim, top_k=RERANK_TOP_K)
            evidence_chunks.extend(local_chunks)

        if instructions.get("search_live"):
            # Fetch from live web / wikipedia
            live_chunks = live_search_service.fetch_live_evidence(claim, max_results=3)
            evidence_chunks.extend(live_chunks)

        return evidence_chunks

    def _judge_evaluate(self, claim: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Judge Agent logic: Evaluates the gathered evidence using LLM or robust Offline NLI.
        Delegates to llm_service to handle deep temporal and entity verification.
        """
        from backend.services.llm_service import llm_service
        
        # We pass the top chunks to the llm_service
        if not chunks:
            return {
                "verdict": "INSUFFICIENT_EVIDENCE",
                "confidence_score": 0.0,
                "explanation": "No verified evidence was found for this claim.",
                "key_reasoning": "Vector index returned 0 relevant chunks.",
                "supporting_chunks": [], 
                "contradicting_chunks": [], 
                "neutral_chunks": [],
                "top_chunks": [],
                "entity_match": "Unverified",
                "temporal_match": "Unverified",
                "country_match": "Not Applicable",
                "role_match": "Not Applicable",
                "llm_provider_used": "LMTA Pipeline"
            }

        llm_result = llm_service.verify_with_llm(claim, chunks)
        
        supporting = []
        contradicting = []
        neutral = []
        for cls in llm_result.get("chunk_classifications", []):
            matched_c = next((c for c in chunks if str(c.get("chunk_id")) == str(cls.get("chunk_id"))), None)
            if matched_c:
                matched_c["stance"] = cls.get("stance", "NEUTRAL")
                matched_c["rationale"] = cls.get("rationale", "")
                if cls.get("stance") == "SUPPORTING":
                    supporting.append(matched_c)
                elif cls.get("stance") == "CONTRADICTING":
                    contradicting.append(matched_c)
                else:
                    neutral.append(matched_c)
        
        return {
            "verdict": llm_result.get("verdict", "INSUFFICIENT_EVIDENCE"),
            "confidence_score": llm_result.get("confidence_score", 0.0),
            "explanation": llm_result.get("explanation", ""),
            "key_reasoning": llm_result.get("key_reasoning", ""),
            "supporting_chunks": supporting,
            "contradicting_chunks": contradicting,
            "neutral_chunks": neutral,
            "top_chunks": chunks,
            "entity_match": llm_result.get("entity_match", "Unverified"),
            "temporal_match": llm_result.get("temporal_match", "Unverified"),
            "country_match": llm_result.get("country_match", "Not Applicable"),
            "role_match": llm_result.get("role_match", "Not Applicable"),
            "llm_provider_used": llm_result.get("llm_provider_used", "LMTA Local Pipeline")
        }

    def process_claim(self, claim: str, context_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Main entry point for the orchestrator.
        """
        # --- VERCEL SERVERLESS FALLBACK ---
        if os.environ.get("VERCEL") == "1":
            instructions = self._router_analyze(claim)
            chunks = self._researcher_gather(claim, instructions)
            return self._judge_evaluate(claim, chunks)
        # ----------------------------------

        instructions = self._router_analyze(claim)
        chunks = self._researcher_gather(claim, instructions)
        
        # Inject the Echo News (Client App) Context directly into the pool with maximum credibility
        if context_text and len(context_text.strip()) > 20:
            chunks.insert(0, {
                "chunk_id": f"client-context-{abs(hash(context_text)) % 100000}",
                "document_name": "Source Article from Echo News",
                "source": "Directly Supplied Context from Client",
                "source_type": "Direct Client Context",
                "text": context_text,
                "credibility_score": 100.0,
                "relevance_score": 100.0,
                "is_live_retrieved": True
            })
            
        # Deduplicate chunks based on text
        seen_texts = set()
        unique_chunks = []
        for c in chunks:
            t = c.get("text", "").strip()
            if t not in seen_texts:
                seen_texts.add(t)
                unique_chunks.append(c)

        # Cross-Encoder Reranking
        from backend.services.rag.local_reranker import local_reranker
        top_chunks = local_reranker.rerank(claim, unique_chunks, top_k=5)
        
        # Filter out clearly irrelevant chunks (MS MARCO logits < -2.0)
        filtered_chunks = [c for c in top_chunks if c.get("relevance_score", -999.0) > -2.0]

        result = self._judge_evaluate(claim, filtered_chunks)
        return result

agent_orchestrator = AgentOrchestrator()
