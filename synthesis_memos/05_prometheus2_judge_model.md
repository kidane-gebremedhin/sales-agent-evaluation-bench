# Synthesis Memo — Prometheus 2: An Open-Source Language Model Specialized in Evaluating Other Language Models

**Paper:** Kim et al. Prometheus 2: An Open-Source Language Model Specialized in Evaluating Other Language Models. 2024.  
**Date:** 2026-04-30 (Act III, Path B).  
**Memo word count:** ~450 words.

## Core Contribution

Prometheus 2 trains a 7B-parameter open-source judge model using a large-scale preference dataset (100K+ feedback instances) to replicate GPT-4-level evaluation capability. The model supports both **direct assessment** (scoring 1–5 on specified criteria) and **pairwise ranking** (choosing between two responses). Training combines fine-grained rubric-based scoring with pairwise comparison objectives.

Key design: the training data includes **rubric text** as part of the input, so the judge learns to condition its scores on the specific evaluation criteria rather than applying a generic quality signal. This makes Prometheus-2 domain-adaptable — supply new rubrics and the judge generalizes.

## What I Agree With — Rubric-Conditioned Judging

This is the core insight for Path B. Our scoring evaluator already defines per-task rubrics with five tone markers (direct, grounded, honest, professional, non-condescending) and deterministic checks. The trained critic we build with SimPO inherits this rubric-conditioned pattern: the preference pairs are constructed against the evaluator's rubric, so the critic learns to score on the same dimensions.

Prometheus-2's finding that rubric-conditioned training outperforms generic "is this good?" training validates our approach of embedding the Tenacious style guide rules into the preference-pair construction rather than relying on the base model's implicit quality sense.

## Where I Disagree — Scale Requirements

Prometheus-2 requires 100K+ training instances. The paper's transfer to new domains is enabled by this massive dataset covering diverse evaluation scenarios. We are building a judge on **~1,000–2,000 preference pairs** in a single narrow domain (B2B sales outreach tone evaluation). The paper does not address whether the rubric-conditioning principle holds at 50× smaller scale.

**My prediction:** At our scale, the rubric signal will need to be encoded more explicitly in the input format rather than learned implicitly from data volume. Our preference pair format will include the tone-marker failure mode as a structured label (e.g., `{"failed_markers": ["honest", "non_condescending"]}`) in the prompt, rather than relying on the model to discover which markers differentiate chosen from rejected.

**Counter-evidence from LIMA (Zhou et al., 2023):** LIMA demonstrated that 1,000 high-quality examples can align a 65B model. Our SimPO-trained critic is 0.8–2B parameters — proportionally, 1,000–2,000 high-quality preference pairs may be sufficient if quality filtering is rigorous.

## Application to Our Path B

We use Prometheus-2's architecture motivation but not its training recipe:
1. **Rubric-conditioned input format:** Each training prompt includes the relevant tone-marker definitions from the style guide.
2. **Scale compensation:** Instead of 100K data points, we compensate with higher preference-pair quality — every rejected example has a documented failure mode; every chosen example passes the full scoring evaluator.
3. **Evaluation:** Our trained critic will be benchmarked against the offline stub judge and the live OpenRouter dev-tier judge on the dev partition, matching Prometheus-2's validation approach of comparing against frontier judge agreement.
