import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.services.verification_service import verification_service
from backend.models.schemas import ClaimVerificationRequest
from backend.api.routes import seed_trusted_knowledge_base
from backend.database.db import init_db

def test_full_pipeline():
    print("==================================================")
    print("Testing Evidence-First Verification Architecture")
    print("==================================================")

    init_db()

    # Step 1: Seed knowledge base
    print("\n[Step 1] Seeding trusted knowledge base...")
    seed_res = seed_trusted_knowledge_base()
    print(f"Seed Result: {seed_res['message']}")

    # Step 2: Test Claim 1: "The current Prime Minister of Japan is Shinzo Abe"
    # Expectation: CONTRADICTED or CURRENT STATUS UNVERIFIED (since he was assassinated, he is no longer current PM)
    claim1 = "The current Prime Minister of Japan is Shinzo Abe"
    print(f"\n[Step 2] Testing Claim 1: \"{claim1}\"")
    req1 = ClaimVerificationRequest(claim=claim1)
    res1 = verification_service.verify(req1)
    print(f"-> Claim Type: {res1.claim_type}")
    print(f"-> Verdict: {res1.verdict}")
    print(f"-> Verification Strength: {res1.verification_strength}")
    print(f"-> Entity Match: {res1.entity_match}")
    print(f"-> Recommendation: {res1.verification_recommendation.what_to_check if res1.verification_recommendation else 'None'}")
    assert res1.verdict in ["CONTRADICTED", "CURRENT STATUS UNVERIFIED", "INSUFFICIENT EVIDENCE", "AMBIGUOUS"], f"Unexpected verdict {res1.verdict}"

    # Step 3: Test Claim 2: "The current Prime Minister of India is Shinzo Abe"
    # Expectation: CONTRADICTED or INSUFFICIENT EVIDENCE (Wrong Country)
    claim2 = "The current Prime Minister of india is Shinzo Abe"
    print(f"\n[Step 3] Testing Claim 2: \"{claim2}\"")
    req2 = ClaimVerificationRequest(claim=claim2)
    res2 = verification_service.verify(req2)
    print(f"-> Verdict: {res2.verdict}")
    print(f"-> Entity Match: {res2.entity_match}")
    assert res2.verdict in ["CONTRADICTED", "INSUFFICIENT EVIDENCE", "AMBIGUOUS"], f"Unexpected verdict {res2.verdict}"

    # Step 4: Test Claim 3: "Global renewable electricity generation surpassed 30% in 2024."
    # Expectation: VERIFIED or HISTORICAL SUPPORT ONLY
    claim3 = "Global renewable electricity generation surpassed 30% in 2024."
    print(f"\n[Step 4] Testing Claim 3: \"{claim3}\"")
    req3 = ClaimVerificationRequest(claim=claim3)
    res3 = verification_service.verify(req3)
    print(f"-> Verdict: {res3.verdict}")
    assert res3.verdict in ["VERIFIED", "HISTORICAL SUPPORT ONLY", "INSUFFICIENT EVIDENCE"], f"Unexpected verdict {res3.verdict}"

    # Step 5: Test Claim 4: "Antibiotics are effective in curing viral infections like the common cold."
    # Expectation: REFUTED (or CONTRADICTED)
    claim4 = "Antibiotics are effective in curing viral infections like the common cold."
    print(f"\n[Step 5] Testing Claim 4: \"{claim4}\"")
    req4 = ClaimVerificationRequest(claim=claim4)
    res4 = verification_service.verify(req4)
    print(f"-> Verdict: {res4.verdict}")
    assert res4.verdict in ["REFUTED", "CONTRADICTED", "CONFLICTING EVIDENCE"], f"Unexpected verdict {res4.verdict}"

    print("\n==================================================")
    print("ALL BACKEND VERIFICATION PIPELINE TESTS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    test_full_pipeline()
