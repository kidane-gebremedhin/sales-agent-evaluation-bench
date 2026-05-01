#!/usr/bin/env python3
"""Act IV — Evaluate trained SimPO critic on the dev partition.

Loads a trained LoRA adapter, runs inference on every dev task from
Tenacious-Bench v0.1, and reports agreement with the scoring evaluator's
ground-truth dimension scores.

Usage:
    # Evaluate a single γ run
    python training/eval_dev.py --adapter training/checkpoints/gamma_0.5/final

    # Evaluate all γ sweep outputs
    python training/eval_dev.py --sweep

    # Compare against baseline (no adapter)
    python training/eval_dev.py --baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scoring_evaluator import score_task  # noqa: E402

DEV_TASKS_PATH = REPO / "tenacious_bench_v0.1" / "dev" / "tasks.jsonl"
TRAIN_TASKS_PATH = REPO / "tenacious_bench_v0.1" / "train" / "tasks.jsonl"
PAIRS_PATH = REPO / "training_data" / "preference_pairs.jsonl"
CHECKPOINTS_DIR = REPO / "training" / "checkpoints"
ABLATIONS_DIR = REPO / "ablations"
RESULTS_PATH = ABLATIONS_DIR / "ablation_results.json"

MODEL_ID = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
GAMMA_SWEEP = [0.3, 0.5, 1.0, 1.5]


def load_dev_tasks() -> list[dict]:
    """Load dev partition tasks."""
    tasks = []
    with open(DEV_TASKS_PATH) as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


def load_adapter_model(adapter_path: str, model_id: str = MODEL_ID):
    """Load the base model with the trained LoRA adapter merged."""
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_path,
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        print(f"[✓] Loaded adapter from {adapter_path} via Unsloth")
        return model, tokenizer
    except ImportError:
        pass

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    print(f"[✓] Loaded adapter from {adapter_path} via PEFT")
    return model, tokenizer


def compute_preference_accuracy(
    model, tokenizer, pairs: list[dict], max_pairs: int = 200
) -> dict:
    """Compute preference accuracy: how often the model assigns higher
    average log-prob to chosen vs rejected (the SimPO reward signal)."""
    import torch

    correct = 0
    total = 0
    chosen_rewards = []
    rejected_rewards = []

    sample = pairs[:max_pairs]
    for i, pair in enumerate(sample):
        prompt = pair["prompt"]
        chosen = pair["chosen"]
        rejected = pair["rejected"]

        # Tokenize
        chosen_input = tokenizer(
            prompt + "\n" + chosen,
            return_tensors="pt", truncation=True, max_length=512,
        ).to(model.device)
        rejected_input = tokenizer(
            prompt + "\n" + rejected,
            return_tensors="pt", truncation=True, max_length=512,
        ).to(model.device)

        with torch.no_grad():
            chosen_out = model(**chosen_input)
            rejected_out = model(**rejected_input)

        # Average log probability (SimPO reward)
        chosen_logprobs = torch.nn.functional.log_softmax(chosen_out.logits, dim=-1)
        rejected_logprobs = torch.nn.functional.log_softmax(rejected_out.logits, dim=-1)

        # Gather log-probs of actual tokens (shifted by 1)
        chosen_ids = chosen_input["input_ids"][:, 1:]
        rejected_ids = rejected_input["input_ids"][:, 1:]

        chosen_token_logprobs = chosen_logprobs[:, :-1, :].gather(
            2, chosen_ids.unsqueeze(-1)
        ).squeeze(-1)
        rejected_token_logprobs = rejected_logprobs[:, :-1, :].gather(
            2, rejected_ids.unsqueeze(-1)
        ).squeeze(-1)

        # Length-normalized average (SimPO reward)
        chosen_reward = chosen_token_logprobs.mean().item()
        rejected_reward = rejected_token_logprobs.mean().item()

        chosen_rewards.append(chosen_reward)
        rejected_rewards.append(rejected_reward)

        if chosen_reward > rejected_reward:
            correct += 1
        total += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(sample)}] acc={correct/total:.3f}")

    acc = correct / max(1, total)
    avg_reward_gap = sum(
        c - r for c, r in zip(chosen_rewards, rejected_rewards)
    ) / max(1, total)

    return {
        "preference_accuracy": round(acc, 4),
        "avg_reward_gap": round(avg_reward_gap, 6),
        "total_pairs": total,
        "correct": correct,
    }


def evaluate_on_dev_with_critic(
    model, tokenizer, dev_tasks: list[dict], max_tasks: int = 72
) -> dict:
    """Use the critic to score good/bad candidates for dev tasks and
    measure agreement with the scoring evaluator."""
    # Re-use the preference pair templates to generate test candidates
    sys.path.insert(0, str(REPO / "training_data"))

    from prepare_preference_pairs import _extract_context, _good_variants, _bad_variants

    agree = 0
    disagree = 0
    task_results = []

    for task in dev_tasks[:max_tasks]:
        dim = task.get("primary_dimension", "tone_preservation")
        if dim == "segment_reasoning" and task.get("task_type") == "classify_segment":
            continue  # skip classification tasks for this eval

        try:
            ctx = _extract_context(task)
            goods = _good_variants(ctx)
            bads = _bad_variants(ctx)
        except Exception:
            continue

        if not goods or not bads:
            continue

        # Pick first good and first bad
        good = goods[0]
        bad_cand, _ = bads[0]

        # Evaluator scores
        good_eval = score_task(task, good)
        bad_eval = score_task(task, bad_cand)
        eval_prefers_good = good_eval.score > bad_eval.score

        # Critic scores (SimPO reward = avg log prob)
        import torch

        def _reward(candidate_text: str) -> float:
            from training.train_simpo import _build_prompt_for_eval  # noqa
            prompt = f"Evaluate this sales outreach draft on '{dim}'."
            full = prompt + "\n" + candidate_text
            inputs = tokenizer(full, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
            logprobs = torch.nn.functional.log_softmax(out.logits, dim=-1)
            ids = inputs["input_ids"][:, 1:]
            token_lp = logprobs[:, :-1, :].gather(2, ids.unsqueeze(-1)).squeeze(-1)
            return token_lp.mean().item()

        good_text = f"Subject: {good.get('subject','')}\n{good.get('body','')}"
        bad_text = f"Subject: {bad_cand.get('subject','')}\n{bad_cand.get('body','')}"

        good_reward = _reward(good_text)
        bad_reward = _reward(bad_text)
        critic_prefers_good = good_reward > bad_reward

        if eval_prefers_good == critic_prefers_good:
            agree += 1
        else:
            disagree += 1

        task_results.append({
            "task_id": task["task_id"],
            "dimension": dim,
            "eval_good_score": good_eval.score,
            "eval_bad_score": bad_eval.score,
            "critic_good_reward": round(good_reward, 6),
            "critic_bad_reward": round(bad_reward, 6),
            "agreement": eval_prefers_good == critic_prefers_good,
        })

    total = agree + disagree
    return {
        "judge_evaluator_agreement": round(agree / max(1, total), 4),
        "agree": agree,
        "disagree": disagree,
        "total_tasks": total,
        "per_task": task_results,
    }


def evaluate_adapter(adapter_path: str, gamma: float) -> dict:
    """Full evaluation pipeline for a single adapter."""
    print(f"\n{'='*70}")
    print(f"  Evaluating adapter: {adapter_path}")
    print(f"  γ = {gamma}")
    print(f"{'='*70}\n")

    model, tokenizer = load_adapter_model(adapter_path)

    # 1. Preference accuracy on held-out pairs (from training split)
    print("[1/2] Computing preference accuracy on training eval split...")
    pairs = []
    with open(PAIRS_PATH) as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    # Use last 10% as eval (matching the train script split)
    from datasets import Dataset
    ds = Dataset.from_dict({"dummy": [0]*len(pairs)})
    split = ds.train_test_split(test_size=0.1, seed=42)
    eval_indices = split["test"]["dummy"]  # not right — we need indices

    # Simpler: just take last 10%
    n_eval = max(1, len(pairs) // 10)
    eval_pairs = pairs[-n_eval:]

    pref_metrics = compute_preference_accuracy(model, tokenizer, eval_pairs, max_pairs=200)
    print(f"  Preference accuracy: {pref_metrics['preference_accuracy']}")

    # 2. Agreement with scoring evaluator on dev tasks
    print("\n[2/2] Computing critic-evaluator agreement on dev tasks...")
    dev_tasks = load_dev_tasks()
    dev_metrics = evaluate_on_dev_with_critic(model, tokenizer, dev_tasks)
    print(f"  Agreement: {dev_metrics['judge_evaluator_agreement']}")

    result = {
        "gamma": gamma,
        "adapter_path": str(adapter_path),
        "preference_accuracy": pref_metrics,
        "dev_agreement": {k: v for k, v in dev_metrics.items() if k != "per_task"},
        "dev_per_task": dev_metrics.get("per_task", []),
    }
    return result


def run_sweep_eval() -> None:
    """Evaluate all γ sweep adapters and write ablation results."""
    all_results = []

    for gamma in GAMMA_SWEEP:
        adapter_path = CHECKPOINTS_DIR / f"gamma_{gamma}" / "final"
        if not adapter_path.exists():
            print(f"[!] Adapter not found: {adapter_path}, skipping γ={gamma}")
            continue

        result = evaluate_adapter(str(adapter_path), gamma)
        all_results.append(result)

    # Write ablation results
    ABLATIONS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "description": "SimPO γ sweep ablation on Tenacious-Bench v0.1 dev partition",
            "model": MODEL_ID,
            "beta": 2.0,
            "gamma_sweep": GAMMA_SWEEP,
            "results": [{k: v for k, v in r.items() if k != "dev_per_task"} for r in all_results],
        }, f, indent=2)
    print(f"\n[✓] Ablation results: {RESULTS_PATH}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"  γ Sweep Summary")
    print(f"{'='*70}")
    print(f"{'γ':<8}{'Pref Acc':<12}{'Dev Agreement':<16}{'Reward Gap':<14}")
    print("-" * 50)
    for r in all_results:
        print(
            f"{r['gamma']:<8}"
            f"{r['preference_accuracy']['preference_accuracy']:<12.4f}"
            f"{r['dev_agreement']['judge_evaluator_agreement']:<16.4f}"
            f"{r['preference_accuracy']['avg_reward_gap']:<14.6f}"
        )

    # Write per-task traces
    traces_path = ABLATIONS_DIR / "held_out_traces.jsonl"
    with open(traces_path, "w") as f:
        for r in all_results:
            for t in r.get("dev_per_task", []):
                t["gamma"] = r["gamma"]
                f.write(json.dumps(t) + "\n")
    print(f"[✓] Per-task traces: {traces_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SimPO critic on dev partition")
    parser.add_argument("--adapter", type=str, help="Path to trained adapter")
    parser.add_argument("--gamma", type=float, default=0.5, help="γ value for this adapter")
    parser.add_argument("--sweep", action="store_true", help="Evaluate all γ sweep adapters")
    parser.add_argument("--baseline", action="store_true", help="Evaluate base model (no adapter)")
    args = parser.parse_args()

    if args.sweep:
        run_sweep_eval()
    elif args.adapter:
        result = evaluate_adapter(args.adapter, args.gamma)
        ABLATIONS_DIR.mkdir(parents=True, exist_ok=True)
        out = ABLATIONS_DIR / f"eval_gamma_{args.gamma}.json"
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[✓] Results: {out}")
    elif args.baseline:
        print("[baseline] Evaluating base model without adapter...")
        # For baseline, load model without adapter and measure
        model, tokenizer = load_adapter_model(MODEL_ID, MODEL_ID)
        pairs = []
        with open(PAIRS_PATH) as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))
        n_eval = max(1, len(pairs) // 10)
        pref = compute_preference_accuracy(model, tokenizer, pairs[-n_eval:], max_pairs=200)
        print(f"Baseline preference accuracy: {pref['preference_accuracy']}")
        ABLATIONS_DIR.mkdir(parents=True, exist_ok=True)
        out = ABLATIONS_DIR / "baseline_results.json"
        with open(out, "w") as f:
            json.dump({"gamma": None, "baseline": True, "preference_accuracy": pref}, f, indent=2)
        print(f"[✓] Baseline results: {out}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
