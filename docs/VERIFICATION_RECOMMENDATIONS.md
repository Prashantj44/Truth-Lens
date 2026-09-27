# Verification Recommendations

TruthLens rejects the "Trust Me" black-box AI model. Instead of generic suggestions ("Search Google"), it provides exact, actionable instructions on how the user can verify a claim themselves.

## Mechanism
The `RecommendationEngine` generates instructions tailored to the parsed `StructuredClaim`.

### Examples
**Claim**: "ABC Institute is located in Mumbai."
- **Source Target**: Official Website / Directory
- **Target Domain**: `.edu, .gov, .org`
- **What to check**: Look for the 'Contact Us' or official address page.
- **What confirms**: The official address exactly matches the location in the claim.
- **What refutes**: The official address points to a different city/country.

**Claim**: "Person X is the current CEO of Company Y."
- **Source Target**: Official corporate leadership page / verified PR.
- **Queries generated**: `"Person X" current role`, `"Person X" replaced`
- **What confirms**: The official source lists Person X as currently holding the role, with no recent announcements of them stepping down.
- **What refutes**: The official source lists someone else in the role, or lists Person X as 'former'.
