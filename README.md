# Sales Agent Evaluation Bench (Tenacious-Bench)

**Week 11 / TRP1 Challenge — Ground Truth.**
*Building the Sales Evaluation Bench and aligning the Conversion Engine.*

> Status: **Act IV (Training) — ready to run**, 2026-04-30.
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

## How to run

### Acts I–III (local, no GPU)

```bash
python scoring_evaluator.py --self-test                # validate rubric on 3 example tasks
python training_data/prepare_preference_pairs.py       # regenerate preference pairs from train partition
python training_data/check_contamination.py            # verify held-out isolation
```

### Act IV — Train, Ablate, and Measure (Google Colab T4 + Unsloth)

> **Runtime:** Free-tier Colab T4 (16 GB VRAM), 4-bit QLoRA via Unsloth.
> Estimated wall-time: ~40-60 min per γ value, ~3-4 hours for the full sweep.

> [!IMPORTANT]
> **Do NOT replace Colab's pre-installed PyTorch.** Unsloth is tightly coupled
> to the torch + CUDA combination that Colab ships. Upgrading/downgrading
> torch is the #1 cause of dependency hell. The install steps below use
> `--no-deps` everywhere to keep the Colab base environment intact.

#### Step 0 — Open Colab and select T4 GPU

Open a **Google Colab** notebook. Set the runtime to **T4 GPU**:
`Runtime → Change runtime type → T4 GPU`.

```python
# Clone the repo (first run only)
!git clone https://github.com/kidane-gebremedhin/sales-agent-evaluation-bench.git
%cd sales-agent-evaluation-bench
```

#### Step 0.5 — Check the Colab environment (diagnostic)

Run this cell **before** installing anything so you know which
torch / CUDA versions Colab currently ships:

```python
import torch, os
print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.version.cuda}")
print(f"GPU      : {torch.cuda.get_device_name(0)}")
print(f"Colab env: {'yes' if 'COLAB_' in ''.join(os.environ) else 'no'}")
```

Expected output (versions will drift over time):

```
PyTorch  : 2.6.0+cu124    # or similar — do NOT change this
CUDA     : 12.4           # or 12.6 / 12.8
GPU      : Tesla T4
Colab env: yes
```

#### Step 1 — Install Unsloth + dependencies (no-deps pattern)

> **Key principle:** Colab's default `transformers`, `trl`, and `datasets` are
> *too new* for Unsloth (May 2026). We downgrade them to the latest versions
> Unsloth supports, upgrade `torchao` (too old), and use `--no-deps` to
> prevent pip from cascading into torch.

```python
# ── 1a. Fix version-incompatible packages Colab ships ──
# These MUST use --no-deps so pip doesn't try to swap out torch.
!pip install --no-deps \
    "transformers==5.5.0" \
    "trl==0.24.0" \
    "datasets==3.6.0" \
    "torchao>=0.13.0"

# ── 1b. Install Unsloth ecosystem (--no-deps) ──
!pip install --no-deps "bitsandbytes>=0.46.1" accelerate xformers peft \
    triton cut_cross_entropy unsloth_zoo sentencepiece protobuf
!pip install --no-deps unsloth

# ── 1c. Install missing deps that Unsloth/unsloth_zoo require ──
!pip install tyro msgspec

# ── 1d. Install remaining project deps (safe to resolve normally) ──
!pip install scikit-learn wandb huggingface_hub hf_transfer
```

> [!TIP]
> If you still see red "ERROR" about dependency conflicts, **ignore them** —
> `--no-deps` intentionally skips resolution. The versions above are tested.

**After this cell finishes → Restart the runtime once:**
`Runtime → Restart session`, then continue from Step 1.1.

#### Step 1.1 — Re-enter repo after restart

```python
%cd /content/sales-agent-evaluation-bench
```

#### Step 1.2 — Verify the install

```python
import torch; print("torch", torch.__version__, "CUDA", torch.version.cuda)

from transformers import __version__ as tv; print("transformers", tv)
from trl import __version__ as trl_v;       print("trl", trl_v)
from peft import __version__ as peft_v;     print("peft", peft_v)

from unsloth import FastLanguageModel;       print("Unsloth ✓")
from trl import CPOTrainer;                  print("CPOTrainer ✓")
```

> [!WARNING]
> If `from unsloth import FastLanguageModel` fails with an import error,
> run the troubleshooting cell below and then restart the runtime again.
>
> ```python
> !pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth
> !pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth_zoo
> ```

#### Step 2 — Dry run (validate config without training)

```python
!python training/train_simpo.py --dry-run
```

Expected output:

```
SimPO training — γ=0.5, β=2.0
Model: unsloth/Qwen3-4B-unsloth-bnb-4bit
LoRA: r=16, α=32
[1/4] Loading preference pairs...
  Total pairs: <from training_data/preference_pairs.jsonl>
  Train: ~90%, Eval: ~10%
[DRY RUN] Would train with the above config. Exiting.
```

#### Step 3 — Train a single γ (quick test)

```python
# Train with default γ=0.5
!WANDB_DISABLED=true python training/train_simpo.py --gamma 0.5
```

The adapter is saved to `training/checkpoints/gamma_0.5/final/`.

#### Step 4 — Full γ sweep (ablation)

```python
# Train all four γ values: 0.3, 0.5, 1.0, 1.5
!WANDB_DISABLED=true python training/train_simpo.py --sweep
```

Or use the shell script:

```python
!WANDB_DISABLED=true bash training/gamma_sweep.sh
```

**Outputs per γ value:**

| Path | Contents |
|---|---|
| `training/checkpoints/gamma_<γ>/final/` | Saved LoRA adapter + tokenizer |
| `training/checkpoints/gamma_<γ>/train_metrics.json` | Loss, runtime, config |
| `training/logs/gamma_sweep_summary.json` | Combined sweep metrics |

#### Step 5 — Evaluate on dev partition

```python
# Evaluate all trained adapters against the scoring evaluator
!python training/eval_dev.py --sweep
```

Or evaluate a single adapter:

```python
!python training/eval_dev.py --adapter training/checkpoints/gamma_0.5/final --gamma 0.5
```

Baseline comparison (no adapter):

```python
!python training/eval_dev.py --baseline
```

**Evaluation outputs:**

| Path | Contents |
|---|---|
| `ablations/ablation_results.json` | γ sweep results (preference accuracy + dev agreement) |
| `ablations/held_out_traces.jsonl` | Per-task critic decisions across all γ values |
| `ablations/eval_gamma_<γ>.json` | Single-adapter evaluation detail |
| `ablations/baseline_results.json` | Base model (no adapter) preference accuracy |

#### Step 6 — Download results

```python
# Zip and download from Colab
!zip -r results.zip training/checkpoints/ ablations/ training/logs/
from google.colab import files
files.download('results.zip')
```

#### Step 7 — Resume quickly after Colab disconnect (optional)

```python
%cd /content/sales-agent-evaluation-bench
!WANDB_DISABLED=true python training/train_simpo.py --resume training/checkpoints/gamma_0.5/checkpoint-200 --gamma 0.5
```

### Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'unsloth'` | Restart runtime after Step 1, then re-run from Step 1.1 |
| `RuntimeError: Torch = X.Y too new` | Unsloth hasn't released a wheel for your torch yet. Pin: `!pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124` then re-run Step 1 |
| `CUDA out of memory` | Reduce `BATCH_SIZE` to 2 in `train_simpo.py`, or reduce `MAX_LENGTH` to 384 |
| xformers / triton errors on T4 | `!pip install --no-deps xformers` (T4 doesn't support Flash Attention 2; Unsloth uses xformers instead) |
| `ValueError: … simpo_gamma` | Upgrade TRL: `!pip install --no-deps trl>=0.12.0` |

### Training configuration reference

| Parameter | Value | Rationale |
|---|---|---|
| Algorithm | SimPO | Reference-free → fits T4 VRAM (no frozen ref model) |
| Backbone | `unsloth/Qwen3-4B-unsloth-bnb-4bit` | Runs on T4 with 4-bit QLoRA + gradient checkpointing |
| LoRA rank / alpha | 16 / 32 | Standard for 1–2B models |
| β (reward scaling) | 2.0 | SimPO recommended default |
| γ (target margin) | sweep: 0.3, 0.5, 1.0, 1.5 | Predicted optimal: 0.3–0.8 for short emails |
| Batch size | 4 × 4 grad accum = 16 eff. | T4 VRAM budget |
| Precision | fp16 + 4-bit QLoRA | Via Unsloth (T4 lacks native bf16; fp16 is used) |
| Epochs | 3 | — |
| Max sequence length | 512 tokens | — |
