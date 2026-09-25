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
        Judge Agent logic: Evaluates the gathered evidence using NLI.
        """
        supporting_chunks = []
        contradicting_chunks = []
        neutral_chunks = []

        chunk_classifications = []

        # Extract core content terms from claim for entity grounding
        stopwords = {'is', 'the', 'of', 'in', 'a', 'an', 'and', 'to', 'are', 'was', 'were', 'it', 'that', 'this', 'for', 'on', 'with', 'at', 'by', 'from', 'be', 'as', 'or', 'who', 'what', 'where', 'when', 'why', 'how'}
        claim_terms = [w.lower().strip("?,.!'\"") for w in claim.split() if w.lower().strip("?,.!'\"") not in stopwords and len(w.strip("?,.!'\"")) > 2]

        for chunk in chunks:
            text = chunk.get("text", "")
            eval_result = nli_evaluator.evaluate_stance(claim, text)
            stance = eval_result["stance"]
            rationale = eval_result["rationale"]

            # Entity Grounding Safeguard: Ensure the chunk shares significant vocabulary with the claim
            if stance in ["CONTRADICTING", "SUPPORTING"] and claim_terms:
                chunk_lower = text.lower()
                overlap_count = sum(1 for term in claim_terms if term in chunk_lower)
                # Require at least 2 overlapping terms or 30% of the claim terms (prevents false matches on single common words)
                if overlap_count < 2 and overlap_count < len(claim_terms) * 0.3:
                    stance = "NEUTRAL"
                    rationale = f"Context lacks sufficient entity overlap ({overlap_count}/{len(claim_terms)} terms); demoted to neutral."

            chunk["stance"] = stance
            chunk_classifications.append({
                "chunk_id": chunk.get("chunk_id", ""),
                "stance": stance,
                "rationale": rationale
            })

            if stance == "SUPPORTING":
                supporting_chunks.append(chunk)
            elif stance == "CONTRADICTING":
                contradicting_chunks.append(chunk)
            else:
                neutral_chunks.append(chunk)

        # Aggregate Verdict
        total_eval = len(supporting_chunks) + len(contradicting_chunks)
        
        if not supporting_chunks and not contradicting_chunks:
            verdict = "INSUFFICIENT EVIDENCE"
            confidence = 25.0
            explanation = "Could not find any concrete evidence supporting or refuting the claim."
            key_reasoning = "NLI model classified all retrieved text as Neutral."
        elif len(contradicting_chunks) > 0 and len(supporting_chunks) == 0:
            verdict = "CONTRADICTED"
            confidence = min(99.0, 80.0 + len(contradicting_chunks) * 5.0)
            explanation = f"The claim is refuted by the retrieved evidence. The most relevant source states: \"{contradicting_chunks[0]['text'][:150]}...\""
            key_reasoning = f"NLI detected {len(contradicting_chunks)} contradictory source(s)."
        elif len(supporting_chunks) > 0 and len(contradicting_chunks) == 0:
            verdict = "SUPPORTED"
            confidence = min(99.0, 80.0 + len(supporting_chunks) * 5.0)
            explanation = f"The claim is corroborated by the retrieved evidence. The primary source confirms: \"{supporting_chunks[0]['text'][:150]}...\""
            key_reasoning = f"NLI detected {len(supporting_chunks)} supporting source(s)."
        else:
            verdict = "MISLEADING"
            confidence = 85.0
            explanation = "Evidence is mixed. The claim may be partially true or lacking context."
            key_reasoning = f"Found {len(supporting_chunks)} supporting and {len(contradicting_chunks)} contradicting sources."

        return {
            "verdict": verdict,
            "confidence_score": round(confidence, 1),
            "explanation": explanation,
            "key_reasoning": key_reasoning,
            "supporting_chunks": supporting_chunks,
            "contradicting_chunks": contradicting_chunks,
            "neutral_chunks": neutral_chunks,
            "chunk_classifications": chunk_classifications,
            "top_chunks": chunks
        }

    def process_claim(self, claim: str, context_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Main entry point for the orchestrator.
        """
        # --- VERCEL SERVERLESS FALLBACK ---
        if os.environ.get("VERCEL") == "1":
            from backend.services.llm_service import llm_service
            # In Vercel, route to cloud LLM APIs or built-in NLI engine
            instructions = self._router_analyze(claim)
            chunks = self._researcher_gather(claim, instructions)
            
            if not chunks:
                return {
                    "verdict": "INSUFFICIENT EVIDENCE",
                    "confidence_score": 0,
                    "explanation": "No verified evidence was found for this claim.",
                    "key_reasoning": "Vector index returned 0 relevant chunks.",
                    "evidence": {"supporting_chunks": [], "contradicting_chunks": [], "neutral_chunks": []},
                    "llm_provider_used": "Vercel API Fallback"
                }
                
            llm_result = llm_service.verify_with_llm(claim, chunks)
            
            supporting = []
            contradicting = []
            neutral = []
            for cls in llm_result.get("chunk_classifications", []):
                matched_c = next((c for c in chunks if c.get("chunk_id") == cls.get("chunk_id")), None)
                if matched_c:
                    if cls.get("stance") == "SUPPORTING":
                        supporting.append(matched_c)
                    elif cls.get("stance") == "CONTRADICTING":
                        contradicting.append(matched_c)
                    else:
                        neutral.append(matched_c)
            if not supporting and not contradicting:
                supporting = chunks[:2]
                
            return {
                "verdict": llm_result.get("verdict", "INSUFFICIENT EVIDENCE"),
                "confidence_score": llm_result.get("confidence_score", 0),
                "explanation": llm_result.get("explanation", ""),
                "key_reasoning": llm_result.get("key_reasoning", ""),
                "evidence": {
                    "supporting_chunks": supporting,
                    "contradicting_chunks": contradicting,
                    "neutral_chunks": neutral
                },
                "top_chunks": chunks,
                "llm_provider_used": llm_result.get("llm_provider_used", "Cloud LLM (Vercel Production)")
            }
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
        result["llm_provider_used"] = "Local NLI + LMTA Agentic Pipeline"
        return result

agent_orchestrator = AgentOrchestrator()
