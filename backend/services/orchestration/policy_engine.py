from typing import Dict, Any, List

class PolicyEngine:
    def __init__(self):
        pass

    def determine_verdict(self, claim_info: dict, evidence_analysis: dict) -> dict:
        """
        Policy Engine deterministically resolves the final verdict from structured evidence analysis.
        evidence_analysis comes from the LLM or NLP service, evaluating entailment.
        """
        verdict = evidence_analysis.get("verdict", "AMBIGUOUS")
        entity_match = evidence_analysis.get("entity_match", "Unverified")
        temporal_match = evidence_analysis.get("temporal_match", "Unverified")
        
        # Override rules based on strict policy
        
        if entity_match == "Mismatch":
            verdict = "INSUFFICIENT EVIDENCE"
            
        if claim_info.get("currentness_required", False):
            if temporal_match == "Outdated":
                verdict = "AMBIGUOUS TIME CONTEXT"
            elif temporal_match == "Mismatch":
                verdict = "CONTRADICTED"
            
        # Ensure we are returning standard taxonomy
        # 'VERIFIED', 'REFUTED', 'CONFLICTING EVIDENCE', 'INSUFFICIENT EVIDENCE', 'HISTORICAL SUPPORT ONLY', 'CURRENT STATUS UNVERIFIED', 'AMBIGUOUS'
        
        # Map legacy or fallback taxonomy to the new explicit policy taxonomy
        mapping = {
            "SUPPORTED": "VERIFIED",
            "SUPPORTED_CURRENT": "VERIFIED",
            "SUPPORTED_HISTORICALLY": "HISTORICAL SUPPORT ONLY",
            "REFUTED": "REFUTED",
            "CONTRADICTED": "REFUTED",
            "AMBIGUOUS_TIME_CONTEXT": "CURRENT STATUS UNVERIFIED",
            "INSUFFICIENT_EVIDENCE": "INSUFFICIENT EVIDENCE",
            "MISLEADING": "AMBIGUOUS"
        }
        
        final_verdict = mapping.get(verdict, verdict)
        
        if not evidence_analysis.get("chunk_classifications"):
            final_verdict = "INSUFFICIENT EVIDENCE"
            
        return {
            "verdict": final_verdict,
            "explanation": evidence_analysis.get("explanation", ""),
            "key_reasoning": evidence_analysis.get("key_reasoning", ""),
            "entity_match": entity_match,
            "temporal_match": temporal_match,
            "country_match": evidence_analysis.get("country_match", "Not Applicable"),
            "role_match": evidence_analysis.get("role_match", "Not Applicable")
        }

policy_engine = PolicyEngine()
