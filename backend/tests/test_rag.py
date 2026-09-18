import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.services.document_processor import process_document
from backend.services.retrieval_service import retrieval_service
from backend.services.verification_service import verification_service
from backend.models.schemas import ClaimVerificationRequest
from backend.api.routes import seed_trusted_knowledge_base
from backend.database.db import init_db, get_analytics

def test_full_pipeline():
    print("==================================================")
    print("Testing TruthLens RAG Fact-Verification Engine")
    print("==================================================")

    init_db()

    # Step 1: Seed knowledge base
    print("\n[Step 1] Seeding trusted knowledge base...")
    seed_res = seed_trusted_knowledge_base()
    print(f"Seed Result: {seed_res['message']}")
    chunk_cnt = retrieval_service.count()
    print(f"Total Chunks in Vector DB: {chunk_cnt}")
    assert chunk_cnt > 0, "Vector DB should have indexed chunks"

    # Step 2: Test Claim 1: "India became the world's third-largest economy in 2025."
    # Expectation: REFUTED or MISLEADING (it is 5th in nominal GDP, projected 3rd in 2027/2028)
    claim1 = "India became the world's third-largest economy in 2025."
    print(f"\n[Step 2] Testing Claim 1: \"{claim1}\"")
    req1 = ClaimVerificationRequest(claim=claim1)
    res1 = verification_service.verify(req1)
    print(f"-> Verdict: {res1.verdict}")
    print(f"-> Confidence: {res1.confidence_score}%")
    print(f"-> Source Credibility: {res1.source_credibility_score}%")
    print(f"-> Agreement Score: {res1.evidence_agreement_score}%")
    print(f"-> Explanation: {res1.explanation}")
    print(f"-> Supporting Chunks: {len(res1.supporting_evidence)}, Contradicting: {len(res1.contradicting_evidence)}")
    assert res1.verdict in ["REFUTED", "MISLEADING"], f"Unexpected verdict {res1.verdict}"

    # Step 3: Test Claim 2: "Global renewable electricity generation surpassed 30% in 2024."
    # Expectation: SUPPORTED
    claim2 = "Global renewable electricity generation surpassed 30% in 2024."
    print(f"\n[Step 3] Testing Claim 2: \"{claim2}\"")
    req2 = ClaimVerificationRequest(claim=claim2)
    res2 = verification_service.verify(req2)
    print(f"-> Verdict: {res2.verdict}")
    print(f"-> Confidence: {res2.confidence_score}%")
    print(f"-> Explanation: {res2.explanation}")
    assert res2.verdict == "SUPPORTED", f"Unexpected verdict {res2.verdict}"

    # Step 4: Test Claim 3: "Antibiotics are effective in curing viral infections like the common cold."
    # Expectation: REFUTED
    claim3 = "Antibiotics are effective in curing viral infections like the common cold."
    print(f"\n[Step 4] Testing Claim 3: \"{claim3}\"")
    req3 = ClaimVerificationRequest(claim=claim3)
    res3 = verification_service.verify(req3)
    print(f"-> Verdict: {res3.verdict}")
    print(f"-> Confidence: {res3.confidence_score}%")
    print(f"-> Explanation: {res3.explanation}")
    assert res3.verdict == "REFUTED", f"Unexpected verdict {res3.verdict}"

    # Step 5: Test Claim 4: "Flying saucers landed in Atlantis in the year 1400."
    # Expectation: INSUFFICIENT EVIDENCE
    claim4 = "Flying saucers landed in Atlantis in the year 1400."
    print(f"\n[Step 5] Testing Claim 4: \"{claim4}\"")
    req4 = ClaimVerificationRequest(claim=claim4)
    res4 = verification_service.verify(req4)
    print(f"-> Verdict: {res4.verdict}")
    print(f"-> Confidence: {res4.confidence_score}%")
    print(f"-> Explanation: {res4.explanation}")
    assert res4.verdict == "INSUFFICIENT EVIDENCE", f"Unexpected verdict {res4.verdict}"

    # Step 6: Test Analytics
    print("\n[Step 6] Testing Analytics...")
    analytics = get_analytics()
    print(f"Total Claims: {analytics['total_claims_verified']}")
    print(f"Verdict Counts: {analytics['verdict_counts']}")
    print(f"Avg Confidence: {analytics['average_confidence']}%")
    assert analytics['total_claims_verified'] >= 4

    print("\n==================================================")
    print("ALL BACKEND VERIFICATION PIPELINE TESTS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    test_full_pipeline()
