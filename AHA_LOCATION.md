# First Aha location

App v2.18 adds a required, initially unselected retrospective item on each formal Part 2 answer page. The three existing Aha sliders remain unchanged. Search is not interrupted. No correctness or intensity threshold gates location reporting.

Choices: before search (zero-query wording refers to entering the answer page), after each actually displayed query result, while answering after search, experienced but cannot locate, and no experience. Query positions use the valid-query history, not rejected attempts. Only the first experience is requested.

Four fields are appended to the existing Google Sheet schema, preserving existing column positions: `aha_location` (before_search / after_feedback / answer_phase / unknown / none), `aha_query_index` (1-based for after_feedback; null otherwise), `aha_location_label`, and `aha_location_measure_version` (retrospective-first-aha-v1-20261111). Existing `app_version` and event timestamps identify collection versions. No exact subjective-onset timestamp is inferred. Earlier records without these fields are missing measurements, not no-Aha responses. Unknown or no-Aha responses do not trigger exclusion.

Location alignment is secondary: compare model progress at the reported feedback position with other eligible feedback positions in the same trial, accounting for feedback level and change, query position, and participant and puzzle dependence. Pre-search, answer-page, unknown, absent, and missing responses are reported separately. They do not enter a query-localization contrast. Model parameters for this comparison use independent updating data and training puzzles only. Existing primary hypotheses and quality rules are unchanged.

## Consistency review in v2.19

A fixed review-only signal marks `aha_location=none` together with an Aha-strength slider score >=80/100 (stored normalized `aha>=0.8`). The rule uses the Aha-strength rating only, not suddenness, surprise or confidence. The 80 threshold is operational and is not a validated exclusion cutoff. Rule version: aha-consistency-v1-20261111. Default slider score 50 does not trigger it.

Trial events append `aha_consistency_flags`, `aha_consistency_review_needed`, `aha_consistency_rule_version`, and `aha_consistency_high_threshold` to the Sheet schema. Completion `quality_summary` includes flagged trial count and identities. These measurement flags are deliberately separate from `quality_flags` and automatic screening: they do not change completion code, quality status, quality pass, payment eligibility, or existing invalidity reasons. No participant-facing warning is shown and no response is forced to agree. Missing historical measurements are not flagged.

The signal alone never excludes a participant or an Aha observation from the primary analysis. Report contradictions and assess sensitivity of Aha-specific secondary analyses to flagged observations. Scientific inclusion and provider acceptance are separate decisions. Under the provider's stated arrangement, a manual rejection after a completion code consumes the 30% rejection allowance; this flag does not turn such a record into a pre-completion automatic rejection. The allowance is not a scientific exclusion target.
