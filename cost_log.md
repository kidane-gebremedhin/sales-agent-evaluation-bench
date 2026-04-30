# Cost Log — Tenacious-Bench (Week 11)

Envelope: **$10 / trainee**. Default training path is free (Unsloth on Colab T4).
Buckets per challenge spec:

| Bucket | Budget | Spent | Remaining |
|---|---|---|---|
| Dataset authoring (dev-tier LLM) | $3–5 | $0.0289 | ~$5.00 |
| Training | $0–5 | $0.00 | $5.00 |
| Held-out evaluation (eval-tier) | $2–3 | $0.00 | $3.00 |
| Reserve | $1–2 | $0.00 | $2.00 |
| **Total** | **$10** | **$0.029** | **$9.97** |

User-set spend cap for Act II: **<$1.00**. Realized: $0.029. ✅

Two non-negotiables (per challenge spec):
- No τ²-Bench retail validation runs this week. Reuse Week 10 score.
- No eval-tier model on Days 2–3. Iteration uses dev-tier exclusively.

## Charges

| Timestamp (UTC) | Bucket | Provider / Model | Purpose | USD |
|---|---|---|---|---|
| 2026-04-29 17:46–17:54 | Dataset authoring | OpenRouter / `deepseek/deepseek-v3.2` and `qwen/qwen3.5-4b-instruct` (rotated) | Multi-LLM synthesis (53 generation calls, 60 attempts incl. retries). See [`generation_scripts/synthesis_log.jsonl`](generation_scripts/synthesis_log.jsonl). | 0.0209 |
| 2026-04-29 17:55 | Dataset authoring | OpenRouter / `deepseek/deepseek-v3.2` | Live judge filter on multi-LLM synthesis pool (53 calls, 26 retained at threshold ≥ 4/5 on each of 3 dimensions). See [`generation_scripts/judge_filter_log.jsonl`](generation_scripts/judge_filter_log.jsonl). | 0.0081 |
| 2026-04-29 (various) | Dataset authoring | local | Trace-derived, programmatic, hand-authored generation; offline stub judge on the three non-LLM pools. | 0.00 |
| **Subtotal Act II** | | | | **0.0289** |

## Per-call cost detail (multi-LLM synthesis)

Recorded by OpenRouter API in the `usage.cost` field; aggregated:

- 60 attempted generations × ~250 tokens in / ~200 out per call
- Mean cost per call: $0.000348
- Total: $0.0209
- 7 calls dropped (json parse / structural validation), 53 retained pre-judge, 26 retained post-judge.

## Per-call cost detail (live judge filter)

- 53 judge calls (one per multi-LLM-synthesis task)
- Mean cost per call: $0.000153
- Total: $0.0081
- 26 of 53 cleared the 4/5 threshold on each of three dimensions.

## Sanity checks

- All API calls capped at `TB_SYNTH_COST_CAP_USD=$0.20` and
  `TB_JUDGE_COST_CAP_USD=$0.30` via guard logic in
  [`generation_scripts/04_multi_llm_synthesis.py`](generation_scripts/04_multi_llm_synthesis.py)
  and [`generation_scripts/judge_filter.py`](generation_scripts/judge_filter.py).
- Neither cap was hit; no cost overruns.
- No τ²-Bench retail re-runs this week. Week 10 score reused as informational reference (`method/method.md` § Delta C).
- No eval-tier model used in Acts I or II.
