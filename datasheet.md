# Datasheet — Tenacious-Bench v0.1

Following Gebru et al., *Datasheets for Datasets* (2021) and Pushkarna
et al., *Data Cards: Purposeful and Transparent Dataset Documentation*
(FAccT 2022). Pushkarna's telescopic/periscopic/microscopic layered
detail is shown inline at the start of each section.

> **Telescopic (one line):** A 237-task, machine-verifiable evaluation
> bench for B2B engineering-talent sales agents — segment classification,
> signal grounding, bench honesty, tone preservation, gap framing —
> built from the Tenacious Conversion Engine (Week 10) traces, public
> Crunchbase ODM and layoffs.fyi, and multi-LLM synthesis.

| | |
|---|---|
| Name | `tenacious_bench_v0.1` |
| Total tasks | 237 (after contamination drops) |
| Partitions | train (119, 50%) · dev (72, 30%) · held_out (46, 19%) |
| Authoring modes | trace_derived (89), programmatic (89), multi_llm_synthesis (26), hand_authored (35) |
| Rubric dimensions | segment_reasoning, signal_grounding, bench_honesty, tone_preservation, gap_framing |
| Difficulty mix | easy 47 · medium 77 · hard 76 · adversarial 37 |
| License | CC-BY-4.0 |
| Maintainer | Kidane (kidane@10academy.org) — TRP1 cohort |
| Date generated | 2026-04-29 |
| Schema | [`schema.json`](schema.json) |
| Scoring evaluator | [`scoring_evaluator.py`](scoring_evaluator.py) |

---

## 1. Motivation (Gebru §1)

> **Periscopic:** τ²-Bench retail does not grade Tenacious-specific
> behavior. Re-running it on Week 10 evidence showed Delta A = −0.033
> (p = 0.742) for the Tenacious mechanism — retail rewards the wrong
> shape of agent. Tenacious-Bench is the bench Tenacious needs and the
> open community currently lacks: graded outreach quality on real
> prospect briefs across five Tenacious-specific failure modes.

**For what purpose was the dataset created?**
To evaluate B2B sales agents (Tenacious's "Conversion Engine" archetype)
on five failure dimensions that public benchmarks miss: ICP segment
reasoning under signal conflict, signal grounding (every claim resolves
to a source), bench honesty (no over-commit), tone preservation across
multi-turn threads, and competitor-gap framing without condescension.

The motivation chain is documented in [`audit_memo.md`](audit_memo.md);
in summary: 34 Week 10 probes pass 100% on scripted invariants but say
nothing about how the agent *reads* on real prospect briefs.

**Who created the dataset?**
A single TRP1 trainee (Kidane), authoring on top of the Week 10
Conversion Engine artifacts and the Week 11 challenge brief.

**Funding?** None. Compute under $0.05 (OpenRouter dev tier).

## 2. Composition (Gebru §2)

> **Periscopic:** Tasks are JSON objects pairing an input
> (hiring-signal brief, prospect metadata, optional competitor-gap
> brief, prior thread, bench summary) with a machine-verifiable rubric.
> 4 authoring modes; 5 rubric dimensions; 6 task types.

> **Microscopic (one task field at a time):**
> - `task_id` (e.g. `TB-2014`) — TB-NNNN, namespaced by mode.
> - `primary_dimension` — the rubric dimension scored.
> - `secondary_dimensions` — additional dimensions weighted lower.
> - `difficulty` ∈ {easy, medium, hard, adversarial}.
> - `source_mode` ∈ {trace_derived, programmatic, multi_llm_synthesis, hand_authored}.
> - `task_type` — classify_segment, compose_cold_1, compose_warm_reply, compose_warm_objection, scheduling_offer, reengagement.
> - `input` — see [`schema.json`](schema.json) for fields.
> - `ground_truth` — populated for classify_segment and bench-honesty handoff tasks; null for open-ended generation tasks scored by rubric only.
> - `rubric` — `deterministic_checks` (regex, predicates, ground-truth match) and `judge_checks` (1–5 marker scores via LLM judge with deterministic stub fallback).
> - `source_provenance` — generator, judge_model, originating_probe_ids, originating_week10_trace_ids, dedup_hash, judge_filter scores.

**What do instances represent?**
Each instance is a single sales-agent task: a structured input plus a
scoring rubric. The agent (or model under test) produces a candidate
output (segment label, email body, or structured handoff decision); the
rubric mechanically returns a score in [0, 1] and a pass/fail.

**How many instances?** 237.

**Sample composition by primary dimension**

| Dimension | n | % |
|---|---:|---:|
| segment_reasoning | 112 | 47.3 |
| bench_honesty     |  52 | 21.9 |
| tone_preservation |  49 | 20.7 |
| signal_grounding  |  18 |  7.6 |
| gap_framing       |   6 |  2.5 |

Gap framing is intentionally small — the failure mode is concentrated
in Segment 4 outreach, which is rare in the Tenacious top-of-funnel
(only 3 of 1,000 Crunchbase prospects classify Segment 4 by the Week 10
classifier). v0.2 will add ≈30 hand-authored gap_framing tasks.

**Sample composition by source mode**

| Source mode | n | % | Cost (USD) |
|---|---:|---:|---:|
| trace_derived | 89 | 37.6 | 0.00 |
| programmatic  | 89 | 37.6 | 0.00 |
| hand_authored | 35 | 14.8 | 0.00 |
| multi_llm_synthesis | 26 | 10.9 | 0.021 |

The synthesis pool was authored by `deepseek/deepseek-v3.2` and
`qwen/qwen3.5-4b-instruct` rotated per call (53 generations,
26 retained after live judge filter). Live judge model:
`deepseek/deepseek-v3.2`. Preference-leakage policy committed in
[`methodology.md`](methodology.md): a model never judges a task it
generated.

**Difficulty distribution**

| Difficulty | n | concentration in held-out |
|---|---:|:---:|
| easy        |  47 | sparse |
| medium      |  77 | sparse |
| hard        |  76 | moderate |
| adversarial |  37 | **concentrated (all 33 hand-authored adversarials)** |

**Is there a label?**
For `segment_reasoning` and `bench_honesty` (handoff branch), yes —
ground_truth is the expected segment label or expected_handoff boolean.
For `compose_*` task types, the label is the rubric verdict, not a
gold output.

**Are there missing values?**
Yes, by design:
- 1 of 5 prospect_meta domains has a fully populated real brief on disk
  (`culcha.com`); the other 89 trace-derived tasks reconstruct the
  brief from Crunchbase + layoffs.csv joins.
- `competitor_gap_brief` is null for all non-Segment-4 tasks.
- `prior_thread` is null for cold-outreach tasks.

**Relationships between instances?**
Programmatic tasks share template scaffolding (sweep IDs in
`source_provenance.sweep_params`). Trace-derived tasks share a real
Week 10 `trp1_week10_conversion_engine_*` trace ID. **Held-out is
sealed against train/dev** — n-gram, embedding, and time-shift checks
all pass (see [`tenacious_bench_v0.1/contamination_check.json`](tenacious_bench_v0.1/contamination_check.json)).

**Recommended splits**

| Split | n | use |
|---|---:|---|
| train      | 119 | preference-pair construction (Path B SimPO) |
| dev        |  72 | iteration during training |
| held_out   |  46 | sealed; eval-tier judge only; release after leaderboard |

**Errors, sources of noise?** Two known sources:
1. Reconstructed briefs (88 tasks) use `hash(domain) % 18` for the
   open-roles count — a deterministic stub, not real Wellfound data.
   Tasks scored on `signal_grounding` should be treated with this
   caveat in mind.
2. The synthesis-LLM rejected 27 of 53 outputs at judge filter; among
   the kept 26, the hardest LLM-authored bench_honesty cluster (TB-30NN
   range) was identified as near-duplicate by contamination_check.py
   and 6 tasks were dropped to seal held-out. Records in
   [`tenacious_bench_v0.1/dropped_for_contamination.json`](tenacious_bench_v0.1/dropped_for_contamination.json).

**Confidential / personal data?** No. All prospect domains are either
synthetic (`synth-NNN.example`, `prospect-NNNN.example`) or already
public via Crunchbase ODM / layoffs.fyi. The single real brief
(`culcha.com`) is on a public domain and contains only signals
derivable from public sources.

## 3. Collection process (Gebru §3)

> **Periscopic:** Four authoring modes ran in parallel. Trace-derived
> mines real Week 10 traces; programmatic sweeps templates over slot
> values that exercise documented probe failure modes; multi-LLM
> synthesis (Magpie-style, Xu et al. 2024) generates hard cases via
> dev-tier LLMs; hand-authored adversarials are written by the
> trainee.

**Mechanism**
Each mode runs as a script in [`generation_scripts/`](generation_scripts/):

| Script | Mode | Output count |
|---|---|---:|
| `01_trace_derived.py` | trace_derived | 90 (89 retained) |
| `02_programmatic.py`  | programmatic  | 92 (89 retained) |
| `03_hand_authored.py` | hand_authored | 35 (35 retained) |
| `04_multi_llm_synthesis.py` | multi_llm_synthesis | 53 (26 retained after judge filter) |

**Sampling strategy**
Trace-derived stratified on prospects whose Crunchbase record has
funding rounds AND a layoff hit (Yellow.ai, Branch, etc.) — the seven
overlap domains carry the highest-rubric-impact tasks. Remainder
randomly sampled from the 1,001-prospect pool with `seed=0`.

**Time frame** Authoring on 2026-04-29 (Day 2). All public source data
was as-of dates committed in each task's `source_provenance`.

**Ethical review** Not formally reviewed; the data sources are public
(layoffs.fyi CC-BY-4.0, Crunchbase ODM under their public sample
license) and no personal identifiers beyond public LinkedIn/team-page
roles are encoded.

## 4. Preprocessing (Gebru §4)

**Cleaning**
- Dedup hash on (`primary_dimension`, `task_type`, `input`,
  `ground_truth`) — sha256:16 prefix; collisions dropped.
- Judge filter: 3-dimension pointwise score (input coherence,
  ground-truth verifiability, rubric-application clarity) on a 1–5
  scale; threshold 4 on each dimension. Live judge for synthesis pool
  (DeepSeek V3.2); offline deterministic stub for the other modes
  (where the rubric and inputs are author-controlled).

**Was the raw data saved?** Yes, the pre-filter pools and per-task
judge scores are committed at
[`tenacious_bench_v0.1/_pool/`](tenacious_bench_v0.1/_pool/) and
[`generation_scripts/judge_filter_log.jsonl`](generation_scripts/judge_filter_log.jsonl).

**Software** Python 3.11; vendored stdlib only (no external deps for
authoring).

## 5. Uses (Gebru §5)

**Has it been used?** No public use yet. The Week 11 ablation in
Act IV will be the first.

**Recommended uses**
- Evaluation of B2B sales agents on Tenacious-style failure modes.
- Training data for preference-tuned judges on outreach quality.
- Source dataset for ablation studies on LIMA-style high-quality
  filtering (the judge filter retains only ~50% of synthesis output).

**Tasks that should NOT use this dataset**
- Generic τ²-Bench-style retail benchmarking — Tenacious-Bench scores
  domain-specific failure modes that do not generalize.
- Production decisions about *real* layoffs/funding — the layoffs.csv
  and Crunchbase ODM excerpts are demonstrations of public signal, not
  authoritative HR data.
- Training to game the rubric — the deterministic predicates are
  publicly documented; a model trained to match them rather than the
  underlying behavior would over-fit.

**Risk mitigation**
The "skeptic's appendix" in the eventual `memo.pdf` documents:
- Public-signal lossiness in `signal_grounding`.
- Subjectivity of `tone_preservation` (closest to the inter-rater
  threshold at 83.3%).
- The kill-switch trigger condition on the trained component.

## 6. Distribution (Gebru §6)

**Will it be distributed?** Yes — pushed to HuggingFace as
`tenacious_bench_v0.1` after Act V. License **CC-BY-4.0**.

**Restrictions** None beyond the license. The held-out partition will
be sealed at publication and released only after a public leaderboard
opens. Sealed-slice tasks are committed to git in `held_out/` so the
authoring is auditable, but the public dataset card will mark them as
"sealed scoring slice — do not train".

## 7. Maintenance (Gebru §7)

**Maintainer** Kidane (kidane@10academy.org).
**Updates** v0.2 planned with: ≈30 more `gap_framing` adversarials, an
expanded synthesis pool with 4-model rotation, real Wellfound
`open_roles_today` for the 88 reconstructed-brief tasks (replaces the
hash-stub), and a real-human inter-rater pass per the publication
checklist.
**Errata** Recorded as GitHub issues against the dataset repo.
**Will old versions remain?** Yes — versions are tagged and the
HuggingFace Hub preserves prior revisions.

---

## Pushkarna telescopic / periscopic / microscopic

| Layer | What it gives a reader |
|---|---|
| **Telescopic** | The single line at the top of this datasheet — what the bench is, in one breath. |
| **Periscopic** | The first paragraph of each numbered section — enough to decide whether to use the bench. |
| **Microscopic** | Per-field documentation in §2 (the `Microscopic` block) and per-script documentation in §3 — enough to reproduce or audit. |

This layering is what Pushkarna et al. argue distinguishes a Data Card
from a flat datasheet: it lets users at three different levels of
investment find what they need without reading the whole document.
