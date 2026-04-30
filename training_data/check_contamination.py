#!/usr/bin/env python3
"""Contamination check: verify training_data/ preference pairs do not leak
into the held-out or dev partitions.

Three checks (per Chen et al., EMNLP 2025):
  1. 8-gram overlap between training pair prompts and held-out/dev task inputs.
  2. Embedding cosine similarity (simple TF-IDF as a cheap proxy).
  3. Task-ID disjointness — no training pair references a held-out or dev task.

Output: training_data/contamination_report.json
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TRAIN_PAIRS = REPO / "training_data" / "preference_pairs.jsonl"
DEV_TASKS = REPO / "tenacious_bench_v0.1" / "dev" / "tasks.jsonl"
HELD_OUT_TASKS = REPO / "tenacious_bench_v0.1" / "held_out" / "tasks.jsonl"
OUT = REPO / "training_data" / "contamination_report.json"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}


def _load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _task_text(task: dict) -> str:
    """Extract text representation of a task for contamination checks."""
    parts = []
    inp = task.get("input", {})
    if isinstance(inp, dict):
        signal = inp.get("hiring_signal_brief", "")
        if isinstance(signal, dict):
            parts.append(json.dumps(signal))
        elif isinstance(signal, str):
            parts.append(signal)
        prospect = inp.get("prospect", {})
        if isinstance(prospect, dict):
            parts.append(prospect.get("company_name", ""))
            parts.append(prospect.get("contact_name", ""))
    parts.append(task.get("task_id", ""))
    return " ".join(str(p) for p in parts)


def check_ngram_overlap(train_texts: list[str], eval_texts: list[str], n: int = 8):
    """Check if any 8-gram from training appears in eval texts."""
    train_ngrams_all: set[tuple[str, ...]] = set()
    for t in train_texts:
        train_ngrams_all.update(_ngrams(_tokenize(t), n))

    violations = []
    for i, et in enumerate(eval_texts):
        eval_ng = _ngrams(_tokenize(et), n)
        overlap = train_ngrams_all & eval_ng
        if overlap:
            violations.append({
                "eval_idx": i,
                "overlapping_ngrams": len(overlap),
                "sample": [" ".join(ng) for ng in list(overlap)[:3]],
            })
    return violations


def check_cosine_similarity(train_texts: list[str], eval_texts: list[str], threshold: float = 0.85):
    """Cheap TF-IDF cosine check (no external dependencies)."""
    import math

    # Build vocabulary
    all_texts = train_texts + eval_texts
    df = Counter()
    for t in all_texts:
        tokens = set(_tokenize(t))
        for tok in tokens:
            df[tok] += 1
    N = len(all_texts)

    def tfidf_vec(text: str) -> dict[str, float]:
        tokens = _tokenize(text)
        tf = Counter(tokens)
        vec = {}
        for tok, count in tf.items():
            idf = math.log((N + 1) / (df.get(tok, 0) + 1))
            vec[tok] = count * idf
        return vec

    def cosine(a: dict, b: dict) -> float:
        keys = set(a) & set(b)
        if not keys:
            return 0.0
        dot = sum(a[k] * b[k] for k in keys)
        na = math.sqrt(sum(v**2 for v in a.values()))
        nb = math.sqrt(sum(v**2 for v in b.values()))
        return dot / (na * nb) if na * nb > 0 else 0.0

    train_vecs = [tfidf_vec(t) for t in train_texts]
    violations = []
    max_sim = 0.0

    for ei, et in enumerate(eval_texts):
        ev = tfidf_vec(et)
        for ti, tv in enumerate(train_vecs):
            sim = cosine(tv, ev)
            max_sim = max(max_sim, sim)
            if sim >= threshold:
                violations.append({
                    "eval_idx": ei,
                    "train_idx": ti,
                    "cosine_similarity": round(sim, 4),
                })
    return violations, round(max_sim, 4)


def check_task_id_disjointness(train_pairs: list[dict], eval_tasks: list[dict]):
    """Ensure no training pair references an eval task ID."""
    eval_ids = {t.get("task_id") for t in eval_tasks}
    violations = []
    for p in train_pairs:
        tid = p.get("task_id")
        if tid in eval_ids:
            violations.append({"task_id": tid, "partition": "eval"})
    return violations


def main():
    print("Contamination check: training_data vs held-out + dev")
    print("=" * 60)

    pairs = _load_jsonl(TRAIN_PAIRS)
    dev = _load_jsonl(DEV_TASKS)
    held = _load_jsonl(HELD_OUT_TASKS)
    eval_tasks = dev + held

    train_texts = [p.get("prompt", "") + " " + p.get("chosen", "") for p in pairs]
    eval_texts = [_task_text(t) for t in eval_tasks]

    # 1. N-gram overlap
    print("\n1. 8-gram overlap check...")
    ng_violations = check_ngram_overlap(train_texts, eval_texts, n=8)
    print(f"   Violations: {len(ng_violations)}")

    # 2. Cosine similarity
    print("2. TF-IDF cosine similarity (threshold 0.85)...")
    cos_violations, max_sim = check_cosine_similarity(train_texts, eval_texts)
    print(f"   Violations: {len(cos_violations)}, max similarity: {max_sim}")

    # 3. Task-ID disjointness
    print("3. Task-ID disjointness...")
    id_violations = check_task_id_disjointness(pairs, eval_tasks)
    print(f"   Violations: {len(id_violations)}")

    passed = (len(ng_violations) == 0 and len(cos_violations) == 0 and len(id_violations) == 0)

    report = {
        "status": "PASS" if passed else "FAIL",
        "training_pairs_count": len(pairs),
        "eval_tasks_count": len(eval_tasks),
        "checks": {
            "ngram_overlap_8": {
                "passed": len(ng_violations) == 0,
                "violation_count": len(ng_violations),
                "violations": ng_violations[:10],
            },
            "cosine_similarity_0.85": {
                "passed": len(cos_violations) == 0,
                "violation_count": len(cos_violations),
                "max_similarity": max_sim,
                "violations": cos_violations[:10],
            },
            "task_id_disjointness": {
                "passed": len(id_violations) == 0,
                "violation_count": len(id_violations),
                "violations": id_violations[:10],
            },
        },
    }

    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nOverall: {'PASS ✓' if passed else 'FAIL ✗'}")
    print(f"Report: {OUT}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
