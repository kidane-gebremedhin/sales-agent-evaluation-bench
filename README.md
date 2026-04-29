# Sales Agent Evaluation Bench (Tenacious-Bench)

**Week 11 / TRP1 Challenge — Ground Truth.**
*Building the Sales Evaluation Bench and aligning the Conversion Engine.*

> Status: **Act I (Audit & Schema Design) — in progress**, 2026-04-29.
> Predecessor: Week 10 Conversion Engine
> ([../conversion-engine](../conversion-engine)).

## What this repo will contain

A 200–300 task evaluation dataset (`tenacious_bench_v0.1/`), a machine-
verifiable scoring evaluator (`scoring_evaluator.py`), a contamination-
checked partition (50/30/20 train/dev/held_out), a small trained
adapter or judge that lifts the Week 10 agent on a Tenacious-specific
failure mode, and the public artifacts: HuggingFace dataset, model card,
and technical blog post.

## Layout

```
audit_memo.md             # Act I: what τ²-Bench retail misses
schema.json               # Tenacious-Bench v0.1 task schema + 3 examples
scoring_evaluator.py      # Machine-verifiable rubric grader
methodology.md            # Path declaration, justification, partitioning
cost_log.md               # Every API and compute charge, with timestamp
tenacious_bench_v0.1/     # Dataset (filled in Act II)
  held_out/, dev/, train/
generation_scripts/       # Authoring code (Act II)
synthesis_memos/          # One-page memos per required reading
training_data/            # Path-formatted training partition (Act III)
training/                 # Training run script and logs (Act IV)
ablations/                # ablation_results.json, held_out_traces.jsonl
method/                   # Method notes, rubric revision history
scripts/                  # Helpers (contamination_check, etc.)
```

## Path declaration (Day 1, preliminary)

**Path B — DPO/SimPO/ORPO a judge or critic.** Justification in
[`methodology.md`](methodology.md).

## Reference baselines

- Week 10 τ²-Bench retail (informational, not re-run this week):
  - `A_baseline` (llm_agent, dev tier): pass@1 = 0.867 (CI 0.733–0.967)
  - `B_mechanism` (dual_control_agent): pass@1 = 0.833 (CI 0.700–0.967)
  - Delta A = −0.033, p = 0.742 → **failed gate** (Week 10 evidence).
- Week 10 production probe library: 34 probes, 12 categories, all 0%
  trigger rate against scripted fixtures. **The gap this benchmark fills
  is graded quality on real prospect briefs, not invariant pass/fail
  on fixtures.**

## How to run (after Act I)

```
make schema-validate       # validate schema.json against scoring_evaluator
make eval-dummy            # run scoring_evaluator.py on the 3 example tasks
```
