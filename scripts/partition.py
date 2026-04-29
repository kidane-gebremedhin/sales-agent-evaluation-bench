"""Stratified partitioning: 50% train / 30% dev / 20% held-out.

Stratification keys (in priority order):
1. primary_dimension — every dimension represented in every partition
2. source_mode — every authoring mode represented in every partition
3. difficulty — adversarial slice present in held_out (originality probe)

The script is deterministic (fixed seed). Held-out partition gets the
hardest tasks: ALL adversarial-difficulty hand_authored tasks go to
held_out (they carry the most originality weight at grading and must
not leak to training).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
POOL_DIR = REPO_ROOT / "tenacious_bench_v0.1" / "_pool"
BENCH_DIR = REPO_ROOT / "tenacious_bench_v0.1"

POOLS = ["trace_derived", "programmatic", "hand_authored", "multi_llm_synthesis"]
PARTITIONS = {"train": 0.50, "dev": 0.30, "held_out": 0.20}


def load_filtered_pools() -> list[dict]:
    out: list[dict] = []
    for name in POOLS:
        p = POOL_DIR / f"{name}_filtered.jsonl"
        if not p.exists():
            print(f"  warn: {p} missing, skipping")
            continue
        for line in p.read_text().splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def stratified_split(tasks: list[dict], seed: int) -> dict[str, list[dict]]:
    """Stratified 50/30/20 split.

    Held-out budget is capped at 20% of total. Per the challenge spec,
    hand_authored adversarial tasks are highest-originality and should
    be over-represented in held-out — they fill the held-out slot
    first, then the remaining held-out budget is filled from the
    hardest non-hand-authored tasks. Train/dev get the rest in 5:3 ratio.
    """
    rng = random.Random(seed)
    total = len(tasks)
    held_target = round(total * PARTITIONS["held_out"])

    # Phase 1 — fill held-out with hand_authored adversarial first, then
    # other adversarial/hard tasks until the budget is met.
    by_priority: dict[int, list[dict]] = defaultdict(list)
    for t in tasks:
        if t["source_mode"] == "hand_authored" and t["difficulty"] == "adversarial":
            by_priority[0].append(t)
        elif t["difficulty"] == "adversarial":
            by_priority[1].append(t)
        elif t["difficulty"] == "hard":
            by_priority[2].append(t)
        elif t["difficulty"] == "medium":
            by_priority[3].append(t)
        else:
            by_priority[4].append(t)
    for k in by_priority:
        rng.shuffle(by_priority[k])

    parts: dict[str, list[dict]] = {p: [] for p in PARTITIONS}
    held_taken: set[str] = set()
    for k in sorted(by_priority):
        for t in by_priority[k]:
            if len(parts["held_out"]) >= held_target:
                break
            parts["held_out"].append(t)
            held_taken.add(t["task_id"])
        if len(parts["held_out"]) >= held_target:
            break

    # Phase 2 — stratified train/dev split on the remaining tasks.
    remaining = [t for t in tasks if t["task_id"] not in held_taken]
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for t in remaining:
        strata[(t["primary_dimension"], t["source_mode"])].append(t)

    train_target = round(total * PARTITIONS["train"])
    # 5:3 ratio for train:dev within remaining.
    for key, items in strata.items():
        rng.shuffle(items)
        n = len(items)
        n_train = round(n * (5 / 8))
        parts["train"].extend(items[:n_train])
        parts["dev"].extend(items[n_train:])

    # Top-up: if rounding left train short of target, transfer dev → train.
    while len(parts["train"]) < train_target and parts["dev"]:
        parts["train"].append(parts["dev"].pop(0))

    return parts


def write_partitions(parts: dict[str, list[dict]]) -> None:
    for name, items in parts.items():
        out_dir = BENCH_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "tasks.jsonl"
        with path.open("w") as f:
            for t in items:
                f.write(json.dumps(t, sort_keys=True) + "\n")


# --------------------------------------------------------------------------
# Contamination-aware dedup swap
# --------------------------------------------------------------------------


def _input_text(t: dict) -> str:
    parts: list[str] = []

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif o is not None:
            parts.append(str(o))

    walk(t.get("input") or {})
    return " ".join(parts)


_TOK_RE = re.compile(r"[A-Za-z0-9_\-]+")


def _tokens(s: str) -> list[str]:
    return [w.lower() for w in _TOK_RE.findall(s)]


def _ngrams(toks: list[str], n: int = 8) -> set[tuple[str, ...]]:
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def _hash_embed(text: str, dim: int = 256) -> list[float]:
    vec = [0.0] * dim
    for tok in _tokens(text):
        h = hash(tok)
        i = h % dim
        vec[i] += 1 if (h >> 16) & 1 else -1
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _contaminated_in_held(parts, cosine_thresh, ngram_n):
    ref_ngrams: set[tuple[str, ...]] = set()
    ref_embeds: list[list[float]] = []
    for t in parts["train"] + parts["dev"]:
        ref_ngrams |= _ngrams(_tokens(_input_text(t)), ngram_n)
        ref_embeds.append(_hash_embed(_input_text(t)))

    contaminated: list[dict] = []
    for t in parts["held_out"]:
        text = _input_text(t)
        held_ngrams = _ngrams(_tokens(text), ngram_n)
        if held_ngrams & ref_ngrams:
            contaminated.append(t); continue
        e = _hash_embed(text)
        if any(_cosine(e, re_v) >= cosine_thresh for re_v in ref_embeds):
            contaminated.append(t)
    return contaminated


def _is_clean_against(t: dict, refs: list[dict], cosine_thresh, ngram_n) -> bool:
    ref_ngrams: set[tuple[str, ...]] = set()
    ref_embeds: list[list[float]] = []
    for r in refs:
        ref_ngrams |= _ngrams(_tokens(_input_text(r)), ngram_n)
        ref_embeds.append(_hash_embed(_input_text(r)))
    text = _input_text(t)
    if _ngrams(_tokens(text), ngram_n) & ref_ngrams:
        return False
    e = _hash_embed(text)
    return not any(_cosine(e, re_v) >= cosine_thresh for re_v in ref_embeds)


def dedup_swap(parts: dict[str, list[dict]], rng: random.Random,
               cosine_thresh: float = 0.85, ngram_n: int = 8) -> dict:
    """Resolve contamination in three escalating steps:

    1. Move contaminated held-out → train.
    2. Pull a clean replacement from train into held-out (must itself be
       clean against train+dev minus the swap).
    3. If no clean replacement exists, DROP the contaminated held-out
       task and the conflicting train task entirely (recorded as
       dropped_for_contamination).
    """
    swapped = 0
    dropped: list[dict] = []
    rounds = 0
    while True:
        rounds += 1
        if rounds > 25:
            break
        contaminated = _contaminated_in_held(parts, cosine_thresh, ngram_n)
        if not contaminated:
            break

        # Step 1+2: try to swap each contaminated held-out task with a
        # clean train candidate.
        for victim in contaminated:
            parts["held_out"].remove(victim)
            # Look for a clean train candidate (preferring hard/adversarial)
            train_sorted = sorted(
                parts["train"],
                key=lambda t: 0 if t["difficulty"] in ("adversarial", "hard") else 1,
            )
            chosen = None
            for cand in train_sorted:
                refs = parts["train"] + parts["dev"] + parts["held_out"]
                # Check candidate is clean against (train + dev) without itself
                test_refs = [t for t in refs if t["task_id"] != cand["task_id"]]
                if _is_clean_against(cand, test_refs, cosine_thresh, ngram_n):
                    chosen = cand
                    break
            if chosen is None:
                # Step 3 — drop both the victim and any one near-duplicate
                # train task to break the cluster.
                dropped.append(victim)
                # Find a near-duplicate in train and drop it too.
                victim_e = _hash_embed(_input_text(victim))
                victim_ng = _ngrams(_tokens(_input_text(victim)), ngram_n)
                for cand in list(parts["train"]):
                    if (_ngrams(_tokens(_input_text(cand)), ngram_n) & victim_ng) or (
                        _cosine(_hash_embed(_input_text(cand)), victim_e) >= cosine_thresh
                    ):
                        parts["train"].remove(cand)
                        dropped.append(cand)
                        break
                continue
            parts["train"].remove(chosen)
            parts["train"].append(victim)
            parts["held_out"].append(chosen)
            swapped += 1

    # Final sweep: any held-out task that is *still* contaminated after
    # the swap loop gets dropped outright. This is rare but handles
    # near-duplicate clusters that span partitions where every candidate
    # in train would itself be a swap-violator. Loop until stable.
    final_drops = 0
    for _ in range(10):
        violators = _contaminated_in_held(parts, cosine_thresh, ngram_n)
        if not violators:
            break
        for v in violators:
            parts["held_out"].remove(v)
            dropped.append(v)
            final_drops += 1

    return {"swapped": swapped, "dropped": len(dropped),
            "dropped_ids": [d["task_id"] for d in dropped], "rounds": rounds,
            "final_sweep_drops": final_drops}


def composition_report(parts: dict[str, list[dict]]) -> dict:
    rows = []
    for name, items in parts.items():
        rows.append({
            "partition": name,
            "n": len(items),
            "by_primary_dimension": dict(Counter(t["primary_dimension"] for t in items)),
            "by_source_mode": dict(Counter(t["source_mode"] for t in items)),
            "by_difficulty": dict(Counter(t["difficulty"] for t in items)),
        })
    return {
        "total": sum(r["n"] for r in rows),
        "partitions": rows,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    p.add_argument("--seed", type=int, default=11_2026)
    p.add_argument("--report", type=Path, default=BENCH_DIR / "composition.json")
    args = p.parse_args()

    tasks = load_filtered_pools()
    if not tasks:
        print("no tasks to partition; abort")
        return 1
    print(f"loaded {len(tasks)} tasks across pools")

    parts = stratified_split(tasks, seed=args.seed)
    swap_stats = dedup_swap(parts, random.Random(args.seed))
    print(f"  dedup_swap: swapped={swap_stats['swapped']} rounds={swap_stats['rounds']} final_sweep_drops={swap_stats['final_sweep_drops']} total_dropped={swap_stats['dropped']}")
    if swap_stats["dropped_ids"]:
        (BENCH_DIR / "dropped_for_contamination.json").write_text(
            json.dumps({"dropped_ids": swap_stats["dropped_ids"]}, indent=2)
        )
    write_partitions(parts)

    report = composition_report(parts)
    args.report.write_text(json.dumps(report, indent=2))
    for r in report["partitions"]:
        print(f"  {r['partition']:<10} n={r['n']:<4} dim={r['by_primary_dimension']}")
        print(f"  {'':<10}      mode={r['by_source_mode']}")
        print(f"  {'':<10}      diff={r['by_difficulty']}")
    print(f"  composition report → {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
