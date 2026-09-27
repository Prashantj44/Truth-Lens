from backend.models.schemas import StructuredClaim

class SourcePlanner:
    def __init__(self):
        pass

    def get_source_strategy(self, claim: StructuredClaim) -> dict:
        """
        Determines the type of sources that would actually verify this claim.
        """
        strategy = {
            "target_domains": [],
            "preferred_source_type": "Primary",
            "search_live": True,
            "search_local": True,
            "required_freshness_days": 365,
            "primary_queries": [],
            "contradiction_queries": []
        }

        if claim.claim_type == "PERSON_ROLE":
            strategy["preferred_source_type"] = "Government / Official"
            strategy["primary_queries"] = [f"{claim.subject} current role", f"{claim.subject} official profile"]
            strategy["contradiction_queries"] = [f"{claim.subject} former", f"{claim.subject} replaced", f"{claim.subject} steps down"]
            if claim.currentness_required:
                strategy["required_freshness_days"] = 30
                
        elif claim.claim_type == "LOCATION":
            strategy["preferred_source_type"] = "Official Website / Directory"
            strategy["primary_queries"] = [f"{claim.subject} official address contact"]
            strategy["target_domains"] = [".edu", ".gov", ".org"]
            
        elif claim.claim_type == "SCIENTIFIC":
            strategy["preferred_source_type"] = "Peer Reviewed / Medical"
            strategy["primary_queries"] = [f"{claim.original_claim} study", f"{claim.subject} efficacy research"]
            
        elif claim.claim_type == "STATISTICAL":
            strategy["preferred_source_type"] = "Original Publisher / Dataset"
            strategy["primary_queries"] = [f"{claim.original_claim} report data"]
            
        else:
            strategy["preferred_source_type"] = "Reputable Secondary"
            strategy["primary_queries"] = [claim.original_claim]

        # Add base query
        if not strategy["primary_queries"]:
            strategy["primary_queries"] = [claim.original_claim]

        return strategy

source_planner = SourcePlanner()
