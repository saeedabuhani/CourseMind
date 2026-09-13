# CourseMind Evaluation Report

- Timestamp: 2026-09-09T09:55:24.242506+00:00
- Model: gpt-4o-mini / text-embedding-3-small
- Reference PDF: JavaScript_חלק_א_מדריך_לימוד.pdf

**Pass rate: 100.0% (13/13)**

| Case | Categories | Result | Reason |
|---|---|---|---|
| EVAL-001 | factual, hebrew, source_tracking | PASS |  |
| EVAL-002 | factual, hebrew | PASS |  |
| EVAL-003 | explanation, hebrew | PASS |  |
| EVAL-004 | multi_chunk, hebrew | PASS |  |
| EVAL-005 | source_tracking | PASS |  |
| EVAL-006 | no_evidence | PASS |  |
| EVAL-007 | no_evidence | PASS |  |
| EVAL-008 | document_isolation_negative | PASS |  |
| EVAL-009 | document_isolation_positive, english | PASS |  |
| EVAL-010 | english | PASS |  |
| EVAL-011 | summary_coverage | PASS |  |
| EVAL-012 | summary_coverage, late_page_coverage | PASS |  |
| EVAL-013 | failure_safety | PASS |  |

## Metrics
- Source hit rate (Hit@K): 5/5
- Grounding refusal success: 2/2
- Document isolation: 2/2
- Summary coverage (early/general): 4/4
- Summary late-page coverage: PASS
