# Evidence Policy

TruthLens uses a strict hierarchical evidence policy to establish ground truth.

## 1. Source Quality Hierarchy
- **Primary / Official**: Direct sources (e.g., `.gov`, `.edu`, official press releases). Treated as ground truth for role and location claims.
- **Reputable Secondary**: Verified news sources (e.g., Reuters, AP). Accepted for events and recent news.
- **Encyclopedic**: (e.g., Wikipedia). Treated as context, NOT as ground-truth for exact current temporal status.

## 2. Entity Matching Rules
To prevent false equivalency (e.g., "Shinzo Abe is PM of India"), the engine mandates:
- The **Subject** must align with the retrieved text.
- The **Role/Predicate** must perfectly align with the subject.
- A contradiction is triggered if the evidence explicitly places the subject in a different role/country.

## 3. Temporal Constraints
- Claims containing "currently", "now", or present-tense verbs triggered `currentness_required = True`.
- Stale evidence for a current claim results in `AMBIGUOUS TIME CONTEXT` or `CURRENT STATUS UNVERIFIED`, never `VERIFIED`.

## 4. Agreement Scoring
- Agreement is strictly based on the number of **independent sources**, NOT the number of chunks retrieved from a single document.
- One single source provides "Corroboration not established", whereas 3 independent sources provide "High agreement".
