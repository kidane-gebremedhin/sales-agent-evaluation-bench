# Act IV — Training the Tenacious Sales Critic

## Overview

This directory contains the SimPO training pipeline for the Tenacious
preference-tuned critic (Path B). The critic is a LoRA-adapted
**qwen/qwen3.5-4b-instruct** trained with **SimPO** (reference-free
preference optimization) to score and reject tonally non-adherent sales
outreach drafts.

## Files

```
training/
├── train_simpo.py       # Main training script (Unsloth + TRL)
├── eval_dev.py          # Evaluation on dev partition + ablations
├── gamma_sweep.sh       # Full γ sweep runner
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── checkpoints/         # Saved adapters (created during training)
│   ├── gamma_0.3/
│   ├── gamma_0.5/
│   ├── gamma_1.0/
│   └── gamma_1.5/
└── logs/                # Training logs
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r training/requirements.txt
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### 2. Single training run (default γ=0.5)

```bash
python training/train_simpo.py
```

### 3. Full γ sweep

```bash
bash training/gamma_sweep.sh
```

### 4. Evaluate

```bash
# Single adapter
python training/eval_dev.py --adapter training/checkpoints/gamma_0.5/final

# All sweep results
python training/eval_dev.py --sweep

# Baseline (no adapter)
python training/eval_dev.py --baseline
```

### 5. Dry run (validate config without GPU)

```bash
python training/train_simpo.py --dry-run
python training/train_simpo.py --sweep --dry-run
```

## Training Configuration

| Parameter | Value | Source |
|---|---|---|
| Algorithm | SimPO (via TRL CPOTrainer) | Meng et al. NeurIPS 2024 |
| Backbone | qwen/qwen3.5-4b-instruct | methodology.md |
| Adapter | LoRA r=16, α=32 | preference_pair_stats.json |
| β (reward scaling) | 2.0 | SimPO recommended |
| γ (target margin) | sweep: 0.3, 0.5, 1.0, 1.5 | synthesis_memos/04 |
| Learning rate | 2e-5 | — |
| Epochs | 3 | — |
| Batch size | 4 × 4 grad accum = 16 eff. | T4 VRAM budget |
| Max length | 512 tokens | — |
| Precision | bf16 + 4-bit base | Unsloth/QLoRA |

## Design Decisions

1. **SimPO over DPO**: Reference-free → no frozen reference model → fits
   T4 VRAM. Length-normalized reward correctly scores short cold emails.

2. **TRL CPOTrainer with `loss_type="simpo"`**: Native SimPO implementation
   in TRL since v0.12. No custom loss code needed.

3. **Unsloth first, PEFT fallback**: Unsloth provides 2× faster training
   on the same T4 hardware. Falls back gracefully if not installed.

4. **γ sweep**: The synthesis memo predicts optimal γ ∈ [0.3, 0.8] for
   our short, format-constrained preference pairs. We sweep [0.3, 0.5,
   1.0, 1.5] to validate.

## Evaluation Metrics

- **Preference accuracy**: How often the critic assigns higher SimPO
  reward (avg log-prob) to chosen vs. rejected.
- **Dev agreement**: Agreement rate between the critic's preference and
  the scoring evaluator on dev-partition task candidates.

Results are written to `ablations/ablation_results.json`.
