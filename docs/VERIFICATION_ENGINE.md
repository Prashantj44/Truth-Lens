# Verification Engine Architecture

TruthLens now implements a modular, Evidence-First Verification Engine designed to evaluate the factual accuracy of claims through structured, defensible reasoning.

## 1. Pipeline Overview
The verification engine is coordinated by the `VerificationOrchestrator` (`backend/services/orchestration/verification_orchestrator.py`).

1. **Claim Parser**: Converts the raw string into a `StructuredClaim`, extracting the subject, claim type (e.g., `PERSON_ROLE`, `LOCATION`), and determining if it requires current evidence (e.g., if it says "is currently").
2. **Source Planner**: Uses the `StructuredClaim` to map to an appropriate evidence retrieval strategy. For instance, `PERSON_ROLE` claims require official government or institutional sources and freshness constraints.
3. **Evidence Retriever**: Queries both local vector databases (Chroma) and live web searches. It uniquely executes both **support queries** and **contradiction queries**.
4. **Entailment Analysis**: Evaluates the retrieved chunks against the claim. It extracts entity matches (to prevent "Shinzo Abe" passing as "PM of India") and temporal context.
5. **Policy Engine**: Deterministically resolves the final verdict using explicit rules. It overrides ambiguous LLM output. For example, if there is an entity mismatch, it strictly enforces `INSUFFICIENT EVIDENCE` or `CONTRADICTED`.
6. **Recommendation Engine**: Generates step-by-step guidance for the user to manually verify the claim, pointing them to a specific source type and search query.

## 2. Core Modules
- `claim_parser.py`: Semantic structuring.
- `source_planner.py`: Defines source requirements.
- `evidence_retriever.py`: Multi-pronged chunk retrieval.
- `policy_engine.py`: Final verdict authority.
- `recommendation_engine.py`: Explainability logic.
