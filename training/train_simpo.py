#!/usr/bin/env python3
"""Act IV — SimPO training with Unsloth + TRL.

Trains a LoRA-adapted Qwen2.5-1.5B-Instruct critic using SimPO on the
preference pairs from Act III.  Designed for a free Colab T4 (16 GB VRAM).

Usage:
    # Default γ=0.5
    python training/train_simpo.py

    # Specific γ for the sweep
    python training/train_simpo.py --gamma 0.3

    # Full γ sweep (convenience)
    python training/train_simpo.py --sweep

    # Resume from checkpoint
    python training/train_simpo.py --resume training/checkpoints/gamma_0.5/checkpoint-200

Environment variables:
    WANDB_PROJECT  — W&B project name (default: tenacious-critic)
    WANDB_DISABLED — set to "true" to skip W&B logging
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
PAIRS_PATH = REPO / "training_data" / "preference_pairs.jsonl"
STATS_PATH = REPO / "training_data" / "preference_pair_stats.json"
TRAIN_TASKS_PATH = REPO / "tenacious_bench_v0.1" / "train" / "tasks.jsonl"
DEV_TASKS_PATH = REPO / "tenacious_bench_v0.1" / "dev" / "tasks.jsonl"
CHECKPOINTS_DIR = REPO / "training" / "checkpoints"
LOGS_DIR = REPO / "training" / "logs"
COST_LOG = REPO / "cost_log.md"

# ---------------------------------------------------------------------------
# Defaults (from preference_pair_stats.json)
# ---------------------------------------------------------------------------
_CONFIG = {"model_id": "Qwen/Qwen2.5-1.5B-Instruct"}
MODEL_ID = _CONFIG["model_id"]
LORA_RANK = 16
LORA_ALPHA = 32
LEARNING_RATE = 2e-5
MAX_LENGTH = 512
NUM_EPOCHS = 3
BETA = 2.0
GAMMA_DEFAULT = 0.5
GAMMA_SWEEP = [0.3, 0.5, 1.0, 1.5]
BATCH_SIZE = 4          # effective batch size on T4
GRAD_ACCUM = 4          # → 16 effective
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
SAVE_STEPS = 100
LOGGING_STEPS = 10


def _log_cost(entry: str) -> None:
    """Append a line to cost_log.md."""
    with open(COST_LOG, "a") as f:
        f.write(f"\n{entry}\n")


def load_preference_dataset(path: Path, max_length: int = MAX_LENGTH):
    """Load preference pairs JSONL into a HuggingFace Dataset."""
    from datasets import Dataset

    prompts, chosens, rejecteds, meta = [], [], [], []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            prompts.append(d["prompt"])
            chosens.append(d["chosen"])
            rejecteds.append(d["rejected"])
            meta.append({
                "task_id": d["task_id"],
                "primary_dimension": d.get("primary_dimension"),
                "score_delta": d.get("score_delta", 0),
            })

    ds = Dataset.from_dict({
        "prompt": prompts,
        "chosen": chosens,
        "rejected": rejecteds,
    })
    return ds


def build_model_and_tokenizer(model_id: str | None = None, use_unsloth: bool = True):
    """Load model with LoRA via Unsloth (preferred) or PEFT fallback."""
    if model_id is None:
        model_id = _CONFIG["model_id"]
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=MAX_LENGTH,
                dtype=None,  # auto-detect
                load_in_4bit=True,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                lora_dropout=0.0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
            print(f"[✓] Loaded {model_id} via Unsloth (4-bit LoRA)")
            return model, tokenizer, "unsloth"
        except ImportError:
            print("[!] Unsloth not available, falling back to PEFT")

    # PEFT fallback
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print(f"[✓] Loaded {model_id} via PEFT (4-bit LoRA)")
    return model, tokenizer, "peft"


def train_single_gamma(
    gamma: float,
    resume_from: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Run a single SimPO training with the given gamma."""
    print(f"\n{'='*70}")
    print(f"  SimPO training — γ={gamma}, β={BETA}")
    print(f"  Model: {_CONFIG['model_id']}")
    print(f"  LoRA: r={LORA_RANK}, α={LORA_ALPHA}")
    print(f"{'='*70}\n")

    # Output dirs
    run_name = f"gamma_{gamma}"
    output_dir = CHECKPOINTS_DIR / run_name
    log_dir = LOGS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Load data (lightweight — works without GPU libs)
    print("[1/4] Loading preference pairs...")
    n_pairs = 0
    with open(PAIRS_PATH) as f:
        for line in f:
            if line.strip():
                n_pairs += 1
    print(f"  Total pairs: {n_pairs}")
    print(f"  Train: ~{int(n_pairs * 0.9)}, Eval: ~{int(n_pairs * 0.1)}")

    if dry_run:
        print("[DRY RUN] Would train with the above config. Exiting.")
        return {"gamma": gamma, "status": "dry_run", "train_size": int(n_pairs * 0.9)}

    # Heavy imports — only when actually training
    from trl import CPOConfig, CPOTrainer

    dataset = load_preference_dataset(PAIRS_PATH)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

    # Load model
    print("[2/4] Loading model...")
    model, tokenizer, backend = build_model_and_tokenizer()

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # TRL's CPOTrainer with loss_type="simpo" implements SimPO
    print("[3/4] Configuring SimPO trainer...")
    training_args = CPOConfig(
        output_dir=str(output_dir),
        logging_dir=str(log_dir),
        run_name=run_name,

        # SimPO-specific
        loss_type="simpo",
        cpo_alpha=0.0,         # pure SimPO (no NLL mixing)

        # SimPO hyperparams
        beta=BETA,
        simpo_gamma=gamma,

        # Training
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_LENGTH // 2,

        # Logging
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,

        # Precision
        bf16=True,
        gradient_checkpointing=True,

        # Misc
        seed=42,
        report_to="wandb" if os.environ.get("WANDB_DISABLED") != "true" else "none",
        remove_unused_columns=False,

        # Resume
        resume_from_checkpoint=resume_from,
    )

    trainer = CPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    # Train
    print("[4/4] Training...")
    t0 = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume_from)
    elapsed = time.time() - t0

    # Save final adapter
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\n[✓] Adapter saved to {final_dir}")

    # Log metrics
    metrics = train_result.metrics
    metrics["gamma"] = gamma
    metrics["beta"] = BETA
    metrics["elapsed_seconds"] = round(elapsed, 1)
    metrics["model"] = MODEL_ID
    metrics["backend"] = backend

    metrics_path = output_dir / "train_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[✓] Metrics saved to {metrics_path}")

    # Cost log
    _log_cost(
        f"| {time.strftime('%Y-%m-%dT%H:%M:%SZ')} | SimPO training γ={gamma} | "
        f"Colab T4 | {elapsed:.0f}s | $0.00 (free tier) |"
    )

    return metrics


def run_sweep(dry_run: bool = False) -> list[dict]:
    """Run the full γ sweep."""
    results = []
    for gamma in GAMMA_SWEEP:
        metrics = train_single_gamma(gamma, dry_run=dry_run)
        results.append(metrics)

    # Summary
    summary_path = LOGS_DIR / "gamma_sweep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[✓] Sweep summary: {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Act IV — SimPO training for Tenacious critic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--gamma", type=float, default=GAMMA_DEFAULT,
        help=f"SimPO target reward margin (default: {GAMMA_DEFAULT})",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help=f"Run full γ sweep: {GAMMA_SWEEP}",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from checkpoint path",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print config and exit without training",
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_ID,
        help=f"Base model (default: {MODEL_ID})",
    )
    args = parser.parse_args()

    _CONFIG["model_id"] = args.model

    if args.sweep:
        run_sweep(dry_run=args.dry_run)
    else:
        train_single_gamma(args.gamma, resume_from=args.resume, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
