import json
import time
import os
import sys

# Ensure backend module can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from backend.services.verification_service import verification_service
from backend.models import ClaimVerificationRequest

def run_benchmark():
    dataset_path = os.path.join(os.path.dirname(__file__), 'benchmark_dataset.json')
    with open(dataset_path, 'r') as f:
        claims = json.load(f)

    correct = 0
    total = len(claims)
    results = []

    print(f"Starting benchmark on {total} claims...\n")
    start_time = time.time()

    for item in claims:
        claim_text = item["claim"]
        expected = item["expected_verdict"]

        req = ClaimVerificationRequest(claim=claim_text)
        try:
            resp = verification_service.verify(req)
            actual = resp.verdict
        except Exception as e:
            actual = f"ERROR: {e}"
            resp = None

        is_correct = (actual == expected)
        if is_correct:
            correct += 1

        print(f"Claim: {claim_text}")
        print(f"Expected: {expected} | Actual: {actual}")
        print(f"Match: {'YES' if is_correct else 'NO'}")
        print("-" * 50)

        results.append({
            "claim": claim_text,
            "expected": expected,
            "actual": actual,
            "correct": is_correct
        })

    elapsed = time.time() - start_time
    accuracy = (correct / total) * 100

    print(f"\n--- Benchmark Results ---")
    print(f"Total Claims: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Time Elapsed: {elapsed:.2f}s")
    print(f"Avg Time per Claim: {elapsed / total:.2f}s")

if __name__ == "__main__":
    run_benchmark()
