# Audit memo — what τ²-Bench retail does not grade about Tenacious

**Date:** 2026-04-29 · **Author:** Kidane (TRP1 Week 11) · **≤600 words.**

## The question

τ²-Bench retail grades a generic dual-control customer-service agent on
30 dev-slice tasks: refunds, cancellations, address updates. Tenacious
runs **outbound B2B engineering-talent sales** — segment classification
on layoff+funding overlap, signal-grounded outreach, bench-to-brief
matching, on-brand tone across 3+ turn threads, scheduling across an
Africa-Europe-US overlap window. None of this is in retail. The Week 10
ablation table makes the gap concrete: the very mechanism Tenacious
needs (`dual_control_agent`, run B) showed Delta A = **−0.033** against
the stock baseline on `retail_dev_30` (CI [−0.167, +0.100], p = 0.742;
runs `act1_baseline_20260428T150035Z_b796dd` and
`act1_baseline_20260428T152217Z_021445`). The Tenacious failure modes
are invisible to the retail scoring surface.

## What Week 10 evidence proves

The Week 10 probe library (34 probes, 12 categories, all 0% trigger
rate per `failure_taxonomy.md` 2026-04-28T10:35Z) demonstrates that
the production agent **clears its scripted invariants** but says
nothing about whether real outreach **reads well to a real CTO**. Five
gaps emerged:

1. **Segment-rule reasoning under signal conflict** — P-0001
   (layoff+funding misclassification) and P-0005 (pure-funding
   Segment 1) anchor the highest-cost failure. The Tenacious target-
   failure-mode memo prices a single P-0001 miss at ≈$75k revenue
   loss + a $50k brand tail. Retail has no analog: there is no
   "post-layoff CFO" who will roast a wrong pitch on LinkedIn.

2. **Bench-to-brief honesty** — P-0201 (Rust ask, zero bench), P-0202
   (fractional CTO ask, one available), P-0203 (data-team
   ambiguity). The bench JSON
   (`bench_summary.json`, 2026-04-21) gates capacity; an agent that
   over-commits to engineers it does not have is a contractual risk.
   Retail's inventory-check tasks measure tool-sequencing, not
   honesty under capacity scarcity.

3. **Tone preservation across multi-turn threads** — P-0301 (marker
   drift over 3-turn thread) and P-0302 (offshore-vendor cliché
   regression). The five Tenacious tone markers (direct, grounded,
   honest, professional, non-condescending) were defined precisely
   because consultants like Andela have trained CTOs to recoil at
   "rockstar" and "world-class." Retail does not grade voice.

4. **Gap framing without condescension** — P-0902 (non-condescending
   Segment 4). Senior engineering leaders know their gaps; the value
   is research specificity, not implication that they are behind.
   This is a paragraph-level rubric, not a tool-call invariant.

5. **Cost-aware abstention** — P-0501 (tone-check regen capped at 1)
   and P-0502 (cost-per-qualified-lead budget). Retail measures
   per-task pass; it does not penalize a 3pp lift bought with 3×
   cost (the Cost-Pareto observable this week grades).

## Trace evidence

Five real Week 10 traces ground the gap:

- `trp1_week10_conversion_engine_b3f5aa034a4f` — enrichment.run on
  `yellow.ai`. The pipeline produces a hiring-signal brief whose
  rubric (segment confidence, AI-maturity score, source-URL
  presence) has no τ² analog.
- `trp1_week10_conversion_engine_a600e0edfc30` — `compose_and_send`
  on `culcha.com`. The hiring brief flags Segment 4 with
  AI-maturity 2; the composer must produce a non-condescending
  research-framed email — this is the Tenacious-Bench
  sweet spot.
- `trp1_week10_conversion_engine_407b0937c82f` — `compose_and_send`
  on `delamode-group.com`. Re-run 19 times across the day; tone
  drift across regenerations is exactly the inconsistency Tenacious-
  Bench must catch.
- `trp1_week10_conversion_engine_3ae86222d486` — τ² baseline run
  (A, llm_agent), 0.867 pass@1.
- `trp1_week10_conversion_engine_1de3d584708e` — τ² mechanism run
  (B, dual_control_agent), 0.833 pass@1. The negative Delta A
  proves retail rewards the wrong shape of agent for Tenacious's
  job.

## Implication

Tenacious-Bench v0.1 must grade segment reasoning, signal grounding,
bench honesty, tone preservation, and gap framing on real-prospect
briefs — five rubric dimensions, machine-verifiable, with adversarial
hand-authored slice. The schema in `schema.json` encodes those five
dimensions; the scoring evaluator in `scoring_evaluator.py` makes them
graded without a human in the loop.
