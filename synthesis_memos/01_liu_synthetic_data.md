# Synthesis memo — Liu et al., *Best Practices on Synthetic Data* (arXiv 2404.07503, COLM 2024)

**Reader:** Kidane (TRP1 Week 11) · **Date:** 2026-04-29 · **Memo length:** ~1 page.

## What the paper argues

Liu, Wei, Liu, and 8 co-authors survey the operational landscape of
synthetic-data generation for LLMs, arguing that the central question
has shifted from "can we generate at scale?" to "can we generate
**responsibly**?" They recommend three quality dimensions —
**factuality, fidelity, unbiasedness** — and pair them with quality
filters, diversity controls, and alignment-aware downstream uses. The
paper is positioned as the operational reference for trainees who
build small bench-quality corpora from a limited seed (which is
exactly Tenacious-Bench's situation).

## What I took from it for Tenacious-Bench

Three direct lifts:

1. **Quality filtering is non-negotiable on multi-LLM synthesis.** I
   judged 53 LLM-authored tasks at a 4/5-on-each-of-three threshold
   and kept 26 — a 49% pass rate that aligns with Liu's published
   filter-attrition numbers for instruction-style data. Without the
   filter, the contamination clusters in `TB-3034`–`TB-3040`
   (bench_honesty near-duplicates) would have entered the dataset.

2. **Diversity through routed multi-LLM generation.** Liu argues a
   single generator family produces a recognizably "shaped"
   distribution; rotation across families is the simplest fix. The
   Tenacious-Bench synthesis pool rotates `deepseek/deepseek-v3.2`
   and `Qwen/Qwen3.5-4b-instruct` per call (`SYNTH_MODELS` in
   [`generation_scripts/04_multi_llm_synthesis.py`](../generation_scripts/04_multi_llm_synthesis.py)).
   The methodology.md commits to never using the same family for
   generation and judging on the same task.

3. **Anchor against ground-truth signal at every step.** Liu's
   "fidelity" axis maps directly to my contamination check —
   `time-shift verification` ensures every public-data reference
   carries an `as_of` timestamp from a documented window. Without
   this, the bench would silently absorb dates from the LLM's
   training distribution.

## Where I disagree

Liu treats **unbiasedness as a co-equal first-class quality
dimension.** For a benchmark like Tenacious-Bench, this is wrong, and
the disagreement is load-bearing.

The paper's framing assumes synthetic data is destined for **training**
a generalist model, where dataset bias propagates into downstream
behavior. Tenacious-Bench is destined for **evaluation**, where the
job of the dataset is to **over-represent the distributions where the
agent fails**. My adversarial slice (37 tasks, all in held_out)
deliberately concentrates layoff+funding overlap, interim-CTO
disqualifiers, and bench-overcommit edge cases. Per Tenacious's own
target-failure-mode memo, the unmitigated trigger rate of P-0001 is
~15–20%; if the bench's prospect distribution mirrored the natural
1,001-prospect Crunchbase sample (where layoff+funding overlap is 7
out of 1,001 = 0.7%), the bench would mis-allocate test signal toward
the easy cases the agent already passes.

Liu's dimensions need a **fourth** for evaluation corpora —
**diagnosticity** — and it should sit *above* unbiasedness in the
ordering. A diagnostic eval set is **biased on purpose** toward the
failure modes the system is most likely to regress on. The paper
acknowledges this implicitly when discussing alignment-test sets but
does not systematize it.

**Concrete cost of accepting Liu's framing as-written:** If I had
sampled the trace-derived pool uniformly from the 1,001 prospects
instead of biasing toward the 7 layoff+funding overlap domains, my
held_out would contain roughly 0–1 P-0001 tasks instead of the 11 I
have. A trained Path B judge would have no signal to learn from on the
single highest-cost failure mode Tenacious sales work exhibits.

## Open question for Day 4

Liu's recommendation to filter on **fidelity** (a stylistic match to
the source distribution) might over-prune adversarial tasks — by
construction, an adversarial task is *less* like the modal Week 10
trace, not more. I will revisit the judge-filter threshold for
adversarial-difficulty tasks specifically; the current 4/5 threshold
on rubric-application clarity may need to drop to 3/5 for hand-
authored adversarials. Hold for inspection during Path B training-data
prep.
