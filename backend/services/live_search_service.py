import re
import json
import urllib.request
import urllib.parse
from typing import List, Dict, Any

class LiveSearchService:
    """
    Real-time Knowledge Grounding and Open Encyclopedia Evidence Retriever.
    Provides dynamic fallback and cross-verification for out-of-vault entities,
    institutions, breaking events, and general factual claims.
    """
    def __init__(self):
        self.headers = {
            'User-Agent': 'TruthLens/2.4 (FactCheckingEngine; research@truthlens.ai; windows)'
        }
        self.common_typo_map = {
            r'\binsititue\b': 'institute',
            r'\binsitute\b': 'institute',
            r'\bcolledge\b': 'college',
            r'\buniveristy\b': 'university',
            r'\bprimeminister\b': 'prime minister',
            r'\bpersident\b': 'president',
            r'\bgoverment\b': 'government',
            r'\btechnolgy\b': 'technology'
        }

    def clean_and_correct_query(self, query: str) -> str:
        corrected = query.strip()
        for typo, fix in self.common_typo_map.items():
            corrected = re.sub(typo, fix, corrected, flags=re.IGNORECASE)
        return corrected

    def extract_search_phrases(self, query: str) -> List[str]:
        cleaned = self.clean_and_correct_query(query)
        phrases = [cleaned]
        
        # Extract potential subject phrase before linking verbs
        match = re.split(r'\b(?:is in|is located in|is the|was the|are in|serves as|became|can cure|cure|causes)\b', cleaned, flags=re.IGNORECASE)
        if len(match) > 1:
            subj = match[0].strip()
            if len(subj) > 3 and subj not in phrases:
                phrases.append(subj)
                
            pred = match[1].strip()
            if len(pred) > 3:
                combined = f"{subj} {pred}".strip()
                if combined not in phrases:
                    phrases.append(combined)

        # Remove filler words
        stopwords = {"is", "are", "was", "were", "in", "on", "at", "the", "a", "an", "of", "and", "or", "to", "for", "by", "with", "from", "that", "this"}
        tokens = [w for w in re.findall(r'\b[a-zA-Z0-9]{3,}\b', cleaned) if w.lower() not in stopwords]
        if tokens:
            kw_phrase = " ".join(tokens)
            if kw_phrase not in phrases:
                phrases.append(kw_phrase)

        return phrases

    def fetch_live_evidence(self, claim: str, max_results: int = 4) -> List[Dict[str, Any]]:
        """
        Dynamically fetches factual evidence chunks from Wikipedia and Open Reference sources.
        """
        search_phrases = self.extract_search_phrases(claim)
        evidence_chunks: List[Dict[str, Any]] = []
        seen_titles = set()

        # 1. Wikipedia OpenSearch & Summary API
        for phrase in search_phrases:
            if len(evidence_chunks) >= max_results:
                break
            try:
                opensearch_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={urllib.parse.quote(phrase)}&limit=3&namespace=0&format=json"
                req = urllib.request.Request(opensearch_url, headers=self.headers)
                with urllib.request.urlopen(req, timeout=4) as response:
                    data = json.loads(response.read().decode('utf-8'))
                    if len(data) >= 2 and data[1]:
                        titles = data[1]
                        urls = data[3] if len(data) >= 4 else []

                        for i, title in enumerate(titles):
                            if title.lower() in seen_titles:
                                continue
                            seen_titles.add(title.lower())

                            # Fetch summary extract
                            encoded_title = urllib.parse.quote(title.replace(' ', '_'))
                            summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_title}"
                            req_sum = urllib.request.Request(summary_url, headers=self.headers)
                            try:
                                with urllib.request.urlopen(req_sum, timeout=4) as s_resp:
                                    s_data = json.loads(s_resp.read().decode('utf-8'))
                                    extract = s_data.get('extract', '')
                                    if extract and len(extract.strip()) > 30:
                                        chunk_id = f"live-wiki-{abs(hash(title)) % 100000}"
                                        page_url = urls[i] if i < len(urls) else f"https://en.wikipedia.org/wiki/{encoded_title}"
                                        
                                        evidence_chunks.append({
                                            "chunk_id": chunk_id,
                                            "document_name": f"Wikipedia: {title}",
                                            "source": f"Wikipedia Global Encyclopedia ({title})",
                                            "source_type": "Encyclopedia / Official Reference",
                                            "page_number": 1,
                                            "text": extract,
                                            "credibility_score": 0.92,
                                            "url": page_url,
                                            "is_live_retrieved": True
                                        })
                                        if len(evidence_chunks) >= max_results:
                                            break
                            except Exception:
                                continue
            except Exception:
                continue

        # 2. DuckDuckGo Instant Answers Fallback
        if not evidence_chunks:
            try:
                clean_q = self.clean_and_correct_query(claim)
                ddg_url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(clean_q)}&format=json&no_html=1&skip_disambig=1"
                req_ddg = urllib.request.Request(ddg_url, headers=self.headers)
                with urllib.request.urlopen(req_ddg, timeout=4) as resp:
                    ddg_data = json.loads(resp.read().decode('utf-8'))
                    abstract = ddg_data.get("AbstractText", "")
                    heading = ddg_data.get("Heading", "Open Web Summary")
                    if abstract:
                        evidence_chunks.append({
                            "chunk_id": f"live-ddg-{abs(hash(heading)) % 100000}",
                            "document_name": f"Web Knowledge Base: {heading}",
                            "source": f"Live Web Knowledge Base ({heading})",
                            "source_type": "Verified Web Reference",
                            "page_number": 1,
                            "text": abstract,
                            "credibility_score": 0.88,
                            "url": ddg_data.get("AbstractURL", ""),
                            "is_live_retrieved": True
                        })
            except Exception:
                pass

        return evidence_chunks

live_search_service = LiveSearchService()
