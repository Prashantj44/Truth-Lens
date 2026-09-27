import re
from backend.models.schemas import StructuredClaim

class ClaimParser:
    def __init__(self):
        # Heuristic rules to extract claim properties
        pass

    def parse(self, text: str) -> StructuredClaim:
        text_lower = text.lower()
        
        # Determine Claim Type
        claim_type = "AMBIGUOUS"
        if "president" in text_lower or "minister" in text_lower or "ceo" in text_lower or "director" in text_lower:
            claim_type = "PERSON_ROLE"
        elif "located in" in text_lower or "headquarters" in text_lower:
            claim_type = "LOCATION"
        elif "percent" in text_lower or "%" in text_lower or "amount" in text_lower:
            claim_type = "STATISTICAL"
        elif "cure" in text_lower or "disease" in text_lower or "infection" in text_lower:
            claim_type = "SCIENTIFIC"

        # Check temporal logic
        currentness_required = False
        time_expression = None
        
        if "is currently" in text_lower or "current" in text_lower or "now" in text_lower or "today" in text_lower:
            currentness_required = True
            time_expression = "present"
            
        # Very basic entity extraction for heuristics (in a real system this would use NER or LLM)
        # We will use simple regex for dates/years as time expressions
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        if year_match:
            time_expression = year_match.group(0)
            if int(time_expression) < 2024:
                claim_type = "HISTORICAL_FACT"
                currentness_required = False
        
        # Simple Subject extraction (assume first few words before is/are/was)
        subject_match = re.search(r'^(.*?)\s+(is|are|was|were|became)\s+', text, re.IGNORECASE)
        subject = subject_match.group(1).strip() if subject_match else None
        
        return StructuredClaim(
            original_claim=text,
            subject=subject,
            claim_type=claim_type,
            time_expression=time_expression,
            currentness_required=currentness_required
        )

claim_parser = ClaimParser()
