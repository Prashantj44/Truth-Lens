import os
import sys

# Simulate Vercel environment
os.environ["VERCEL"] = "1"

from starlette.testclient import TestClient
from api.index import app

def run_tests():
    client = TestClient(app)
    passed = 0
    total = 0

    def check(name, condition, extra=""):
        nonlocal passed, total
        total += 1
        if condition:
            passed += 1
            print(f"[PASS] {name} {extra}")
        else:
            print(f"[FAIL] {name} {extra}")

    print("=" * 60)
    print("TruthLens Deep Backend Verification Test Suite (Vercel Mode)")
    print("=" * 60)

    # 1. Health check
    r = client.get("/api/health")
    check("Health Check Status 200", r.status_code == 200)
    check("Health Check Returns Chunks", r.json().get("indexed_chunks", 0) > 0, f"({r.json().get('indexed_chunks')} chunks)")

    # 2. News Status (Knowledge Vault feed)
    r = client.get("/api/news/status")
    check("News Status 200", r.status_code == 200)
    data = r.json()
    check("Total Live Articles > 0", data.get("total_live_articles", 0) > 0, f"({data.get('total_live_articles')} articles)")
    check("Recent Articles List Populated", len(data.get("recent_articles", [])) > 0, f"({len(data.get('recent_articles', []))} stories)")
    check("Active Feeds Present", len(data.get("active_feeds", [])) >= 4, f"({len(data.get('active_feeds', []))} feeds)")

    # 3. Trending Claims
    r = client.get("/api/news/trending")
    check("Trending Claims 200", r.status_code == 200)
    check("Trending List Populated", isinstance(r.json(), list) and len(r.json()) > 0, f"({len(r.json())} trending items)")

    # 4. History
    r = client.get("/api/history")
    check("History Endpoint 200", r.status_code == 200)

    # 5. Analytics
    r = client.get("/api/analytics")
    check("Analytics Endpoint 200", r.status_code == 200)

    # 6. Verification: Truthful Claim
    r = client.post("/api/verify", json={"claim": "The earth revolves around the sun"})
    check("Verify POST 200", r.status_code == 200, f"status={r.status_code}")
    if r.status_code == 200:
        v_data = r.json()
        check("Verdict in Response", v_data.get("verdict") in ["SUPPORTED", "REFUTED", "MISLEADING", "INSUFFICIENT EVIDENCE"], f"Verdict={v_data.get('verdict')}")
        check("Confidence Score Valid", 0 <= v_data.get("confidence_score", -1) <= 100, f"Score={v_data.get('confidence_score')}")

    # 7. Verification: Contradictory Claim
    r = client.post("/api/verify", json={"claim": "Donald Trump is the Prime Minister of India"})
    check("Verify False Claim POST 200", r.status_code == 200, f"status={r.status_code}")
    if r.status_code == 200:
        v_data = r.json()
        check("Verdict Generated", bool(v_data.get("verdict")), f"Verdict={v_data.get('verdict')}")

    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed.")
    print("=" * 60)

    if passed == total:
        print("ALL BACKEND TESTS PASSED SUCCESSFULLY!")
        return 0
    return 1

if __name__ == "__main__":
    sys.exit(run_tests())
