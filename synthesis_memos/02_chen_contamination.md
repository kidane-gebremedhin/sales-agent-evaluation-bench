# Synthesis memo — Chen et al., *Recent Advances in LLM Benchmarks against Data Contamination: From Static to Dynamic Evaluation* (arXiv 2502.17521, EMNLP 2025)

**Reader:** Kidane (TRP1 Week 11) · **Date:** 2026-04-29 · **Memo length:** ~1 page.

## What the paper argues

Chen et al. survey contamination-resistant evaluation, mapping the
field's trajectory from **static** (fixed corpora, vulnerable to
training-set leakage as web crawls catch up) to **dynamic** (re-generated
or perturbed at evaluation time) benchmarks. Their central claim is
that dynamic evaluation is the only sustainable defense, but that the
field currently lacks shared design principles for what makes a
dynamic benchmark *rigorous* (as opposed to merely fresh). They
propose a framework of design principles and maintain a curated
GitHub list of contemporary benchmarks of both kinds.

## What I took from it for Tenacious-Bench

Three concrete operational lifts:

1. **Multi-channel contamination checks, not just n-gram.** Chen
   argues (correctly) that an 8-gram check is necessary but
   insufficient — the LLM that generated a synthesis task may have
   produced a paraphrase that shares no 8-gram with any train task
   while still being effectively the same task. The
   [`scripts/contamination_check.py`](../scripts/contamination_check.py)
   accordingly runs **three** checks: n-gram (≥8 contiguous tokens),
   embedding cosine (< 0.85 against any reference), and time-shift
   (held-out tasks reference time windows distinct from train/dev).
   Without the embedding check, the `TB-30NN` near-duplicate cluster
   would have passed: those tasks share <8-gram overlap but cosine
   similarity > 0.90.

2. **The held-out partition is the only one that needs to be sealed.**
   Chen's framing distinguishes between static-public (`dev`),
   static-private (`held_out`), and dynamic-regenerated. I made the
   spec's three-way split (50/30/20) but committed to releasing
   `held_out/` only after a public leaderboard exists — which matches
   Chen's recommendation for the "private static" partition lifecycle.

3. **Document the contamination report as a first-class artifact.** I
   commit `tenacious_bench_v0.1/contamination_check.json` and
   `dropped_for_contamination.json` so any future re-build can verify
   that 6 specific task IDs were dropped to seal the held-out
   partition. The paper makes this explicit as a "transparency for
   future replicators" requirement.

## Where I disagree

Chen et al. argue **dynamic evaluation is strictly preferable to
static** because it is contamination-immune by construction. Two
problems with this for Tenacious-Bench:

1. **Dynamic benchmarks are non-comparable across model versions.**
   If every evaluation pass re-generates the held-out partition, a
   model improvement of +3pp on Tuesday's slice tells me nothing
   about whether it would have improved Wednesday's. For a
   small-budget engagement like this one (the Week 11 ablation must
   produce a publishable Delta A with 95% CI on a *single*
   sealed slice), dynamic regeneration multiplies the variance by
   the regeneration noise. My partition stays static.

2. **Contamination is a much smaller risk for small, recent
   benchmarks than the paper implies.** Chen's framing was honed on
   benchmarks like MMLU and HumanEval, both of which predate the
   training cutoffs of every modern frontier model. Tenacious-Bench
   was authored on 2026-04-29, after the publication cutoff of the
   models I am evaluating against (Claude Opus 4.7, January 2026
   knowledge cutoff). Contamination risk on this specific bench is
   structurally low; the embedding-check (intended to catch
   intra-bench leaks between train and held-out, not internet
   contamination) is doing the load-bearing work.

**Concrete consequence:** I am NOT regenerating held-out at every eval
run, contra Chen's strong dynamic-preferred recommendation. I AM
running all three contamination checks before sealing, which is the
static-with-discipline middle ground the paper somewhat dismisses. The
right next-step (Week 12 or v0.2) is *partial* dynamic regeneration —
keep the rubric and prospect-meta fixed, but re-generate the brief
text via paraphrase model swaps. That gives reproducibility within a
release while resisting cross-release leakage.

## Open question for Act V

Chen flags that **a public leaderboard accelerates contamination** —
once the held-out gets quoted on Twitter, every model trained
afterward has indirect access. This is the strongest argument for
*never* fully releasing `held_out/`. I will commit to a hash-only
release policy for held-out (publish task hashes, accept evaluation
runs via PR with the runner's logs, score privately) — a
hybrid Chen et al. mention but do not develop.
