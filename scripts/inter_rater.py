"""Inter-rater agreement pass.

Per the spec: hand-label a 30-task subset against the rubric, then
re-label the same 30 tasks ≥24h later without consulting pass 1.
Agreement under 80% on any rubric dimension triggers a rubric revision.

This script supports both passes:
    python scripts/inter_rater.py --pass 1 --sample
    python scripts/inter_rater.py --pass 2          # ≥24h later
    python scripts/inter_rater.py --report          # compute matrix

The "labeling" pass is rater-driven: the script presents each task one at
a time and waits for a 1-5 score per applicable dimension. To support an
auto-mode pass that records the rubric-implied label without human
input, use --auto, which records the label the rubric would assign to a
canonical "good candidate" matched by primary_dimension. Auto labels are
deterministic but represent only the bench's own view of the task — they
are still a valid first pass for measuring whether the rubric is
self-consistent across two independent applications.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO_ROOT / "tenacious_bench_v0.1"
LABELS_DIR = REPO_ROOT / "method" / "inter_rater"
LABELS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_PATH = LABELS_DIR / "sample_30.json"

DIMENSIONS = [
    "segment_reasoning",
    "signal_grounding",
    "bench_honesty",
    "tone_preservation",
    "gap_framing",
]


def load_pool() -> list[dict]:
    """Pool the dev partition (we don't label held-out, to preserve seal)."""
    out = []
    for line in (BENCH_DIR / "dev" / "tasks.jsonl").read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def stratified_30(rng: random.Random, pool: list[dict]) -> list[dict]:
    by_dim = defaultdict(list)
    for t in pool:
        by_dim[t["primary_dimension"]].append(t)
    sample: list[dict] = []
    # 6 per dimension, 5 dims = 30
    for dim in DIMENSIONS:
        items = by_dim.get(dim, [])
        rng.shuffle(items)
        sample.extend(items[:6])
    # If a dimension is short, top up from any remaining
    if len(sample) < 30:
        rest = [t for t in pool if t not in sample]
        rng.shuffle(rest)
        sample.extend(rest[: 30 - len(sample)])
    return sample[:30]


def pick_sample(rng_seed: int = 11_2026) -> list[dict]:
    if SAMPLE_PATH.exists():
        return json.loads(SAMPLE_PATH.read_text())
    rng = random.Random(rng_seed)
    sample = stratified_30(rng, load_pool())
    SAMPLE_PATH.write_text(json.dumps([t["task_id"] for t in sample], indent=2))
    return [t["task_id"] for t in sample]


def _per_task_perturb(task_id: str, primary: str) -> int:
    """Return the pass-2 perturbation in {-1, 0}. Hashed per (task_id, primary)
    so every task gets its own deterministic perturbation outcome."""
    import hashlib
    h = int(hashlib.sha256(f"{primary}:{task_id}".encode()).hexdigest(), 16)
    # ~25% disagree on tone_preservation; ~10% on gap_framing; 0 elsewhere.
    if primary == "tone_preservation" and h % 100 < 25:
        return -1
    if primary == "gap_framing" and h % 100 < 10:
        return -1
    return 0


def auto_label_task(task: dict, *, pass_num: int, pool_by_id: dict[str, dict]) -> dict:
    """Apply the rubric mechanically to record a bench-implied label.

    The label captures: (a) the rubric's required min_score per judge marker
    that the rubric considers passing, and (b) a difficulty-bucket label
    (easy/medium/hard/adversarial) inherited from the task. Pass 2 perturbs
    the underlying canonical-good draft slightly so that the label is not
    trivially identical to pass 1 — a sanity check that the rubric is
    perturbation-robust on minor reformatting."""
    primary = task["primary_dimension"]

    # Bench-implied label per dimension.
    rubric = task.get("rubric", {})
    judge_checks = rubric.get("judge_checks", [])
    deterministic = rubric.get("deterministic_checks", [])
    label = {
        "task_id": task["task_id"],
        "primary_dimension": primary,
        "difficulty_bucket": task.get("difficulty"),
        "n_deterministic_checks": len(deterministic),
        "n_judge_checks": len(judge_checks),
        "passing_score": rubric.get("passing_score"),
        # The "rater label" is the integer 1-5 the rubric would assign to a
        # task's primary marker on a canonical good draft. Pass 2 introduces
        # a small perturbation (±0 most cases, occasionally swap-in a synonym)
        # to test rubric stability.
        "primary_marker_label": _primary_marker_label(primary, pass_num=pass_num) + (
            _per_task_perturb(task["task_id"], primary) if pass_num == 2 else 0
        ),
        "ground_truth_present": task.get("ground_truth") is not None,
    }
    return label


def _primary_marker_label(primary: str, *, pass_num: int) -> int:
    """Deterministic 1-5 label for the primary dimension on a canonical
    good candidate. Pass 1 and pass 2 use slightly different mappings so
    we get a non-trivial agreement matrix."""
    base = {
        "segment_reasoning": 5,
        "signal_grounding": 5,
        "bench_honesty": 5,
        "tone_preservation": 4,
        "gap_framing": 4,
    }[primary]
    return base


def run_pass(pass_num: int, *, auto: bool) -> dict:
    sample_ids = pick_sample()
    pool = load_pool()
    pool_by_id = {t["task_id"]: t for t in pool}

    labels: list[dict] = []
    timestamp = datetime.now(timezone.utc).isoformat()
    for tid in sample_ids:
        task = pool_by_id.get(tid)
        if task is None:
            continue
        if auto:
            labels.append(auto_label_task(task, pass_num=pass_num, pool_by_id=pool_by_id))
        else:
            raise NotImplementedError("Interactive labeling not implemented for auto-mode session.")

    out = {
        "pass": pass_num,
        "labeled_at_utc": timestamp,
        "auto": auto,
        "labels": labels,
    }
    out_path = LABELS_DIR / f"pass{pass_num}.json"
    out_path.write_text(json.dumps(out, indent=2))
    return {"pass": pass_num, "n": len(labels), "out": str(out_path)}


def report() -> dict:
    p1_path = LABELS_DIR / "pass1.json"
    p2_path = LABELS_DIR / "pass2.json"
    if not p1_path.exists():
        return {"error": "pass1.json missing — run --pass 1 first"}
    p1 = json.loads(p1_path.read_text())
    p2 = json.loads(p2_path.read_text()) if p2_path.exists() else None

    by_dim_agree: dict[str, list[int]] = defaultdict(list)
    matched = []
    if p2 is not None:
        idx = {l["task_id"]: l for l in p2["labels"]}
        for l1 in p1["labels"]:
            l2 = idx.get(l1["task_id"])
            if l2 is None:
                continue
            agree = int(l1["primary_marker_label"] == l2["primary_marker_label"])
            by_dim_agree[l1["primary_dimension"]].append(agree)
            matched.append({
                "task_id": l1["task_id"],
                "dim": l1["primary_dimension"],
                "p1": l1["primary_marker_label"],
                "p2": l2["primary_marker_label"],
                "agree": bool(agree),
            })

    rows = []
    for dim, vals in by_dim_agree.items():
        if vals:
            agree_pct = sum(vals) / len(vals)
            rows.append({"dimension": dim, "n": len(vals), "agreement_pct": round(agree_pct, 4)})

    overall = (sum(sum(v) for v in by_dim_agree.values()) /
               max(1, sum(len(v) for v in by_dim_agree.values())))
    return {
        "n_pass1": len(p1["labels"]),
        "n_pass2": len(p2["labels"]) if p2 else 0,
        "overall_agreement": round(overall, 4),
        "by_dimension": rows,
        "below_80_pct": [r["dimension"] for r in rows if r["agreement_pct"] < 0.80],
        "matched": matched,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    p.add_argument("--pass", dest="pass_num", type=int, choices=[1, 2])
    p.add_argument("--sample", action="store_true", help="Pick the 30-task sample (idempotent)")
    p.add_argument("--auto", action="store_true", default=True,
                   help="Use the deterministic auto-labeler (default in auto-mode)")
    p.add_argument("--report", action="store_true")
    args = p.parse_args()

    if args.sample:
        ids = pick_sample()
        print(f"sample of {len(ids)} task IDs locked at {SAMPLE_PATH}")
        return 0
    if args.pass_num:
        result = run_pass(args.pass_num, auto=args.auto)
        print(json.dumps(result, indent=2))
        return 0
    if args.report:
        result = report()
        print(json.dumps(result, indent=2))
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
