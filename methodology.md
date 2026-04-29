# Methodology — Tenacious-Bench v0.1

**Status:** Day 1 draft. Path declared. Partition protocol committed.
**Date:** 2026-04-29.

## Path declaration — Path B (preference-tuned judge or critic)

I am taking **Path B**. The trained component will be a small
preference-scored critic (Qwen 3.5 0.8B or 2B with LoRA), trained
with **SimPO** as the algorithm of choice (justification below). The
critic will be deployed as a rejection-sampling layer in front of the
Week 10 composer's existing tone-preservation check — it scores agent
drafts on the five Tenacious markers plus the three signal-grounded
dimensions (`bench_honesty`, `signal_grounding`, `gap_framing`) and
either accepts the draft or returns the lower-scoring one for rerun.

### Why Path B (and not Path A or C)

The Week 10 evidence points to **inconsistency**, not generation
quality and not trajectory failures:

- The 34-probe library shows **0% trigger rate across all categories**
  (`probes/failure_taxonomy.md`, snapshot 2026-04-28T10:35:28Z). The
  agent passes its scripted invariants. So the failure mode is not
  "the agent generates bad text" — it is "the agent cannot tell when
  a borderline draft is good enough to send."
- Trace `trp1_week10_conversion_engine_407b0937c82f` (compose_and_send
  on `delamode-group.com`) was re-run **19 times during 2026-04-27**
  (07:26Z through 18:01Z). Multiple re-runs of the same prospect
  generation indicate the operator iterating on a draft the agent
  itself could not score reliably — the textbook inconsistency
  signature for Path B.
- Trace `trp1_week10_conversion_engine_a600e0edfc30` (compose_and_send
  on `culcha.com`, Segment 4 with AI-maturity 2) is the gap-framing
  failure mode that judge-based rejection sampling is designed for —
  the composer must produce a non-condescending research-framed email,
  and the current single-LLM tone check sometimes accepts subtly
  condescending drafts.
- The Week 10 ablation (`method/ablation_results.json`,
  2026-04-28T16:19Z) shows Delta A = **−0.033** for the Week 10
  `dual_control_agent` mechanism on `retail_dev_30` (p = 0.742, gate
  failed). That mechanism layered prompt rules. Path A would layer
  more prompt rules; Path B replaces the rule layer with a learned
  preference scorer that does what the rules try and fail to do.

Path A teaches generation skills the agent already has (the disallowed-
phrases regex and the 120-word body limit are deterministic). Path C
addresses trajectory failures, which Tenacious-Bench v0.1 will largely
not exhibit because its scoring is per-message, not per-trajectory
(addressed in the Skeptic's Appendix as a v0.2 expansion).

### Why SimPO over DPO and ORPO

- **SimPO** (Meng, Xia, Chen, NeurIPS 2024) is reference-free, halving
  GPU memory versus DPO at the same batch size — material on a Colab
  T4 with the Qwen 3.5 2B base. The implicit reward is length-
  normalized log-prob, which avoids DPO's documented length bias on
  short, format-constrained generations like a 120-word cold email.
- **DPO** carries the reference model in memory and on every step;
  rejected for cost.
- **ORPO** (Hong, Lee, Thorne, EMNLP 2024) is a strong runner-up but
  fuses preference and SFT objectives, which is harder to ablate when
  the goal is to isolate the preference signal from the base model's
  prior. SimPO gives a cleaner Delta B story.

The path-specific synthesis memo on SimPO will revisit this on Day 4
in light of the actual training data shape.

## Cited Week 10 evidence

| Trace ID | What it shows |
|---|---|
| `trp1_week10_conversion_engine_b3f5aa034a4f` | Enrichment pipeline producing a hiring-signal brief — the input shape Tenacious-Bench tasks consume. |
| `trp1_week10_conversion_engine_407b0937c82f` | Repeated compose_and_send on the same prospect — operator could not rely on the agent's self-judgment. |
| `trp1_week10_conversion_engine_a600e0edfc30` | Segment 4 / AI-maturity 2 compose — exact gap-framing failure mode Path B addresses. |
| `trp1_week10_conversion_engine_3ae86222d486` | τ²-Bench retail A_baseline (0.867 pass@1) — Week 10 reference. |
| `trp1_week10_conversion_engine_1de3d584708e` | τ²-Bench retail B_mechanism (0.833 pass@1) — Week 10 mechanism *lost* on retail; rule-layering is not the answer. |

## Partition protocol

| Partition | Share | Purpose |
|---|---|---|
| `train/` | 50% | Used to construct preference pairs for SimPO. |
| `dev/`   | 30% | Public dev slice — released alongside the schema for community reproduction. |
| `held_out/` | 20% | Sealed; eval-tier judge only; released only after the leaderboard is public. |

### Contamination prevention (Chen et al., EMNLP 2025)

Three checks run before any task enters `held_out/`. Output committed
to `tenacious_bench_v0.1/contamination_check.json`.

1. **N-gram overlap** — fewer than 8 contiguous tokens shared between
   any held-out task input and any train/dev task input.
2. **Embedding cosine** — held-out vs. all train/dev pairs, threshold
   < 0.85, embeddings via `intfloat/e5-small-v2` (cheap, free).
3. **Time-shift verification** — every public-data reference (layoffs,
   funding) carries an `as_of` timestamp; held-out tasks reference
   windows disjoint from train/dev windows.

The contamination-check script lives at
[`scripts/contamination_check.py`](scripts/contamination_check.py)
(implemented in Act II) and runs as part of the dataset publication
pipeline.

## Authoring-mode quotas (per challenge spec)

| Mode | Target share |
|---|---|
| Trace-derived | ~30% |
| Programmatic with parameter sweeps | ~30% |
| Multi-LLM synthesis (Magpie-style) | ~25% |
| Hand-authored adversarial | ~15% |

Per-task metadata records the `source_mode`, generator-model,
judge-model, and judge-score so the `composition.csv` rollup is
reproducible.

## Preference-leakage prevention (Li et al., 2025)

Rotation policy committed before Day 2 authoring begins:

| Role | Models in rotation |
|---|---|
| Hard-seed authoring (frontier) | Claude Sonnet 4.6 OR GPT-class (one frontier model used for ≤ 50 seeds). |
| Bulk-variation generator (dev) | Qwen3-Next-80B-A3B and DeepSeek V3.2 (rotated per-task). |
| Quality-filter judge | A model NOT used to generate that task. Prometheus-2 7B (open) for high-volume; eval-tier reserved for the calibration spot-check. |

Never use the same model family to generate and judge the same task.
Authoring metadata records both choices so leakage is auditable.

## Scoring evaluator status

Implemented in
[`scoring_evaluator.py`](scoring_evaluator.py). Self-test
(`python scoring_evaluator.py --self-test`) runs the three example
tasks against canned good/bad candidate pairs; all three discriminate
good from bad. The judge predicate uses an offline deterministic stub
unless `TB_USE_LIVE_JUDGE=1` is set, in which case it routes to a
dev-tier OpenRouter model.

## Inter-rater agreement plan

Day 3 — hand-label a 30-task subset against the rubric. Day 4 — re-
label the same 30 tasks without looking at the first labels. If
agreement is below 80% on any rubric dimension, revise the rubric and
re-label. Matrix committed to `inter_rater_agreement.md`.

## Open items

- Datasheet draft (Day 3, per Gebru and Pushkarna).
- `composition.csv` rollup script.
- Final SimPO pick reaffirmed against ORPO once training-data shape is
  known (Day 4).
