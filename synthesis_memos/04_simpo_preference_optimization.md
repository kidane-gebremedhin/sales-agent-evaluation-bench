# Synthesis Memo — SimPO: Simple Preference Optimization with a Reference-Free Reward

**Paper:** Meng, Xia, & Chen. SimPO: Simple Preference Optimization with a Reference-Free Reward. NeurIPS 2024.  
**Date:** 2026-04-30 (Act III, Path B).  
**Memo word count:** ~500 words.

## Core Contribution

SimPO replaces DPO's implicit reward (log-ratio between policy and reference model) with the **average log probability of the sequence** under the policy alone, eliminating the reference model entirely. It adds a **target reward margin** γ to the Bradley-Terry objective to enforce a minimum gap between winning and losing responses:

$$L_\text{SimPO}(\pi_\theta) = -\mathbb{E}\left[\log \sigma\left(\frac{\beta}{|y_w|}\log\pi_\theta(y_w|x) - \frac{\beta}{|y_l|}\log\pi_\theta(y_l|x) - \gamma\right)\right]$$

Two key design properties:
1. **Length normalization** — dividing by `|y|` prevents bias toward longer sequences.
2. **Reference-free** — halves GPU memory because no frozen reference model is loaded.

SimPO outperforms DPO by up to 6.4 points on AlpacaEval 2 and 7.5 points on Arena-Hard. It also outperforms ORPO consistently across Mistral-7B and Llama-3-8B setups.

## What I Agree With — and Why It Fits Our Path B

The length-normalization design is critical for our use case. Tenacious cold outreach is constrained to ≤120 words. DPO's unnormalized log-probability reward would penalize short, well-formed emails relative to longer, verbose ones — precisely the anti-pattern the style guide bans. SimPO's average-log-probability reward correctly scores a terse, signal-dense 89-word cold email (GOOD #1 in the style guide) higher than a 152-word self-promotion wall (BAD #1).

The reference-free property is a hard constraint: on a free Colab T4 with 16 GB VRAM, loading Qwen/Qwen3.5-4b-instruct in 16-bit LoRA plus a frozen reference copy would not fit. SimPO's elimination of the reference model is what makes our training budget ($0 compute) feasible.

## Where I Disagree — Target Margin Sensitivity

The paper recommends γ between 0.5 and 1.5 based on general chat benchmarks (AlpacaEval 2, MT-Bench, Arena-Hard). Our preference pairs are **not** general chat — they are short, format-constrained sales drafts where the difference between chosen and rejected is often a single tone violation (e.g., the word "bench" used externally, or an assertive claim on a weak signal). The reward gap between a GOOD #5 draft ("I cannot tell from the outside...") and a BAD #2 draft ("you are scaling aggressively") is subtle in log-probability space.

**My prediction:** the optimal γ for our domain is lower than SimPO's general recommendation — likely 0.3–0.8. A γ of 1.5 would push the model to treat subtle tone violations as catastrophic reward drops, risking overcorrection where the judge rejects all borderline drafts. I will run a small γ sweep {0.3, 0.5, 1.0, 1.5} on the dev partition and report in ablation_results.json.

**Evidence against the paper's sweep:** the paper's experiments use UltraFeedback preference pairs averaging 200+ tokens per response. Our chosen/rejected pairs average ~90 tokens. Shorter sequences have less log-probability mass, so the absolute reward difference is already smaller — adding a high margin compounds the problem.

## Application to Tenacious-Bench Training Data

For our preference pairs, I construct:
- **Chosen:** Corrected outputs that pass the scoring evaluator (all five tone markers ≥ 4/5) plus deterministic checks.
- **Rejected:** Probe-triggered failures from the training partition — outputs that fail one or more evaluator dimensions.
- **Preference-leakage prevention:** Chosen rewrites use DeepSeek V3.2; the judge filter uses Qwen/Qwen3.5-4b-instruct (different model family, per Li et al. 2025).

β is set to 2.0 initially (SimPO's recommended range), with the γ sweep above.
