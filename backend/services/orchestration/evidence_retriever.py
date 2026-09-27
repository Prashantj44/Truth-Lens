from typing import List, Dict, Any
from backend.services.retrieval_service import retrieval_service
from backend.services.live_search_service import live_search_service
from backend.config import RERANK_TOP_K

class EvidenceRetriever:
    def __init__(self):
        pass

    def retrieve(self, claim: str, strategy: dict) -> List[Dict[str, Any]]:
        evidence_chunks = []
        
        # We search the original claim
        if strategy.get("search_local"):
            local_chunks = retrieval_service.retrieve(claim, top_k=RERANK_TOP_K)
            evidence_chunks.extend(local_chunks)

        if strategy.get("search_live"):
            live_chunks = live_search_service.fetch_live_evidence(claim, max_results=3)
            evidence_chunks.extend(live_chunks)
            
            # Also search for contradictions if specified
            for cq in strategy.get("contradiction_queries", []):
                 contra_chunks = live_search_service.fetch_live_evidence(cq, max_results=2)
                 for c in contra_chunks:
                     c["support_type"] = "Potential Contradiction"
                 evidence_chunks.extend(contra_chunks)

        # Deduplicate chunks based on text
        seen_texts = set()
        unique_chunks = []
        for c in evidence_chunks:
            t = c.get("text", "").strip()
            if t not in seen_texts:
                seen_texts.add(t)
                unique_chunks.append(c)

        # Basic Reranking (Cross-Encoder)
        from backend.services.rag.local_reranker import local_reranker
        top_chunks = local_reranker.rerank(claim, unique_chunks, top_k=8)
        
        # Filter out clearly irrelevant chunks (MS MARCO logits < -2.0)
        filtered_chunks = [c for c in top_chunks if c.get("relevance_score", -999.0) > -2.0]
        
        return filtered_chunks

evidence_retriever = EvidenceRetriever()
