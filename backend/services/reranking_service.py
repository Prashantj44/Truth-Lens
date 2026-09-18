import re
from typing import List, Dict, Any
from backend.config import RERANK_TOP_K, SOURCE_CREDIBILITY_WEIGHTS

class RerankingService:
    def rerank(
        self,
        query: str,
        candidate_chunks: List[Dict[str, Any]],
        top_k: int = RERANK_TOP_K
    ) -> List[Dict[str, Any]]:
        """
        Reranks candidate retrieved chunks using a hybrid scoring model:
        1. Semantic vector similarity (50%)
        2. Lexical keyword overlap / BM25 proxy (30%)
        3. Factual entity/numerical match bonus (20%)
        Also computes and attaches source credibility.
        """
        if not candidate_chunks:
            return []

        # Extract meaningful terms and numerical entities from claim
        query_terms = self._tokenize(query)
        query_entities = self._extract_key_entities(query)

        reranked = []
        for chunk in candidate_chunks:
            text = chunk.get("text", "")
            chunk_terms = self._tokenize(text)
            
            # 1. Base semantic score
            semantic_score = float(chunk.get("relevance_score", 0.5))

            # 2. Lexical overlap score (Jaccard / term match)
            if query_terms and chunk_terms:
                common_terms = query_terms.intersection(chunk_terms)
                lexical_score = len(common_terms) / len(query_terms)
            else:
                lexical_score = 0.0

            # 3. Entity & numerical match bonus
            entity_score = 0.0
            if query_entities:
                matched_entities = [e for e in query_entities if e.lower() in text.lower()]
                entity_score = len(matched_entities) / len(query_entities)

            # Composite rerank score
            composite_score = (
                0.50 * semantic_score +
                0.30 * lexical_score +
                0.20 * entity_score
            )
            composite_score = round(min(1.0, max(0.0, composite_score)), 4)

            # Assign source credibility based on source_type
            source_type = chunk.get("source_type", "General").lower()
            credibility = 0.65
            for key, weight in SOURCE_CREDIBILITY_WEIGHTS.items():
                if key in source_type:
                    credibility = weight
                    break

            reranked_chunk = dict(chunk)
            reranked_chunk["relevance_score"] = composite_score
            reranked_chunk["credibility_score"] = credibility
            reranked.append(reranked_chunk)

        # Sort descending by composite relevance score
        reranked.sort(key=lambda x: x["relevance_score"], reverse=True)
        return reranked[:top_k]

    def _tokenize(self, text: str) -> set:
        """Tokenizes text into lowercase alpha-numeric words, filtering stop words."""
        stopwords = {
            "the", "a", "an", "and", "or", "in", "on", "at", "to", "for",
            "is", "are", "was", "were", "of", "with", "by", "that", "this",
            "it", "as", "from", "be", "has", "have", "had", "its", "will"
        }
        words = re.findall(r"\b\w+\b", text.lower())
        return {w for w in words if w not in stopwords and len(w) > 2}

    def _extract_key_entities(self, text: str) -> List[str]:
        """Extracts numbers, years, percentages, and capitalized proper nouns."""
        entities = []
        # Numbers, dates, years, percentages (e.g. 2025, 3rd, 50%, $10B)
        numerics = re.findall(r"\b(?:\d{4}|\d+(?:\.\d+)?%?|\d+(?:st|nd|rd|th)|third|first|second)\b", text, re.IGNORECASE)
        entities.extend(numerics)
        
        # Proper nouns (words capitalized in non-start positions or distinctive)
        capitalized = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
        entities.extend(capitalized)
        
        return list(set(entities))

reranking_service = RerankingService()
