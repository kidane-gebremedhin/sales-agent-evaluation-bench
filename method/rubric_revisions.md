# Inter-Rater Rubric Revision History

*Tracking changes to the `scoring_evaluator.py` criteria based on human inter-rater calibration passes.*

## Iteration 1: Baseline Inter-Rater Agreement Drift
- **Date:** 2026-04-29
- **Issue:** During initial offline `pass1.json` evaluation for the `tone_preservation` dimension, two independent human raters achieved only **83.3% agreement**. 
- **Cause:** Rater A systematically passed templates invoking the phrase "world-class AI solutions", assuming generic professionalism. Rater B explicitly failed them, citing the Tenacious Style Guide's directive to avoid hyperbolic "vendor-speak" in cold CTO outreach.
- **Evidence file:** `method/inter_rater/pass1.json`

## Iteration 2: Deterministic Anti-Marker Enforcement
- **Date:** 2026-04-30 (Act II Integration)
- **Resolution:** The subjective divergence necessitated structurally tightening the offline logic array in `scoring_evaluator.py`. We explicitly coded deterministic *anti-markers* acting as direct vetoes for hyperbolic language.
- **Code implementation:** Added the `_DISALLOWED_PHRASES` array to the `stub_marker_score` function:
  ```python
  _DISALLOWED_PHRASES = [
      r"\btop talent\b",
      r"\bworld[- ]class\b",
      r"\bA[- ]players?\b",
      r"\brockstars?\b",
      r"\bninjas?\b"
  ]
  ```
- **Validation:** Executing the newly enforced array over `pass2.json` raised human-to-evaluator inter-rater alignment on `tone_preservation` from 83.3% up to 95.7%. By mathematically defining "condescending vendor-speak", we eliminated the subjective boundary.

## Pending (v0.2 Roadmap)
- Incorporate pairwise LLM routing directly into `pass3` checks for tie-breaking edge cases involving contextual "directness" vs "abruptness".
