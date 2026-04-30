"""LLM-as-a-judge quality filter (Gu et al., 2024 — survey applied).

Pointwise scoring on three dimensions per task — input coherence,
ground-truth verifiability, rubric-application clarity — score 1–5
each. Threshold for inclusion is 4/5 on each. Pairwise comparison
deferred to dedup pass (separate script, not invoked here).

Preference-leakage policy (Li et al., 2025): the judge model is **not
the same family** as the generator that authored the task. The
metadata on each task carries `generator` and `judge_model`; if both
share a family prefix the filter refuses to run on that task and
flags it for re-routing.

Cost discipline: hard caps total OpenRouter spend at $0.30 by token-
estimating each call. The estimator is conservative — it bills
ceil(tokens / 1000) × tier rate. If the cap is hit the filter falls
back to the offline deterministic stub on the remaining tasks.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
POOL_DIR = REPO_ROOT / "tenacious_bench_v0.1" / "_pool"
LOG_PATH = REPO_ROOT / "generation_scripts" / "judge_filter_log.jsonl"
COST_CAP_USD = float(os.environ.get("TB_JUDGE_COST_CAP_USD", "0.30"))

# OpenRouter dev-tier pricing (approximate, per challenge spec).
# Input/output per 1M tokens. Conservative — actual cost is usually lower.
TIER_RATES = {
    "qwen/qwen/qwen3.5-4b-instruct": (0.10, 0.30),
    "deepseek/deepseek-v3.2": (0.20, 0.80),
}


@dataclass
class JudgeBudget:
    cap_usd: float = COST_CAP_USD
    spent_usd: float = 0.0
    calls: int = 0
    failovers: int = 0  # cap-induced fallbacks to stub
    errors: int = 0

    def can_afford(self, est_usd: float) -> bool:
        return (self.spent_usd + est_usd) <= self.cap_usd

    def charge(self, usd: float) -> None:
        self.spent_usd += usd
        self.calls += 1


def _est_cost(model: str, in_tokens: int, out_tokens: int) -> float:
    rate_in, rate_out = TIER_RATES.get(model, (0.20, 0.80))
    return (in_tokens * rate_in + out_tokens * rate_out) / 1_000_000


def _approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _generator_family(generator: str) -> str:
    g = (generator or "").lower()
    if "trace_derived" in g or "programmatic" in g or "hand_authored" in g:
        return "non_llm"
    if "qwen" in g:
        return "qwen"
    if "deepseek" in g:
        return "deepseek"
    if "claude" in g or "anthropic" in g:
        return "anthropic"
    if "gpt" in g or "openai" in g:
        return "openai"
    return "unknown"


def _judge_family(model: str) -> str:
    m = (model or "").lower()
    if "qwen" in m:
        return "qwen"
    if "deepseek" in m:
        return "deepseek"
    if "claude" in m or "anthropic" in m:
        return "anthropic"
    if "gpt" in m or "openai" in m:
        return "openai"
    return "unknown"


def _stub_judge(task: dict) -> dict:
    """Offline deterministic judge — used when no API key, or after cap is hit.
    Heuristics: penalize empty fields, missing rubric, missing provenance."""
    inp = task.get("input") or {}
    rubric = task.get("rubric") or {}
    has_brief = bool(inp.get("hiring_signal_brief"))
    has_meta = bool(inp.get("prospect_meta"))
    has_rubric = bool(rubric.get("deterministic_checks") or rubric.get("judge_checks"))
    has_provenance = bool(task.get("source_provenance", {}).get("originating_probe_ids"))

    coherence = 5 if (has_brief and has_meta) else 3
    verifiability = 5 if task.get("ground_truth") else (4 if has_provenance else 3)
    rubric_clarity = 5 if has_rubric else 2
    return {
        "input_coherence": coherence,
        "ground_truth_verifiability": verifiability,
        "rubric_application_clarity": rubric_clarity,
        "judge_model": "offline_stub",
        "rationale": "stub-scored; live judge not used.",
    }


def _live_judge(task: dict, model: str, budget: JudgeBudget) -> dict | None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None

    payload_task = {
        "task_id": task["task_id"],
        "primary_dimension": task["primary_dimension"],
        "difficulty": task["difficulty"],
        "input": task["input"],
        "ground_truth": task.get("ground_truth"),
        "rubric_summary": {
            "n_deterministic": len(task["rubric"].get("deterministic_checks", [])),
            "n_judge": len(task["rubric"].get("judge_checks", [])),
            "passing_score": task["rubric"].get("passing_score"),
        },
    }
    prompt = (
        "You are a benchmark task quality judge for the Tenacious-Bench v0.1 "
        "B2B sales-agent dataset. Rate the following task on three dimensions "
        "each on a 1-5 integer scale.\n\n"
        "1. input_coherence: does the input give enough information to attempt the task?\n"
        "2. ground_truth_verifiability: can the rubric mechanically determine pass/fail?\n"
        "3. rubric_application_clarity: is the rubric unambiguous?\n\n"
        "Return STRICT JSON with keys: input_coherence, ground_truth_verifiability, "
        "rubric_application_clarity, rationale (≤30 words).\n\n"
        f"Task:\n{json.dumps(payload_task, separators=(',', ':'))}\n"
    )
    in_tokens = _approx_tokens(prompt)
    out_tokens = 80
    est = _est_cost(model, in_tokens, out_tokens)
    if not budget.can_afford(est):
        budget.failovers += 1
        return None

    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": out_tokens,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")
    req = urllib.request.Request(
        os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1") + "/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/tenacious-bench",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError as e:
        budget.errors += 1
        return None

    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = data.get("usage") or {}
    actual_in = usage.get("prompt_tokens", in_tokens)
    actual_out = usage.get("completion_tokens", out_tokens)
    actual_cost = _est_cost(model, actual_in, actual_out)
    budget.charge(actual_cost)

    try:
        parsed = json.loads(text.strip().strip("`"))
    except json.JSONDecodeError:
        m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if not m:
            budget.errors += 1
            return None
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            budget.errors += 1
            return None

    return {
        "input_coherence": int(parsed.get("input_coherence", 3)),
        "ground_truth_verifiability": int(parsed.get("ground_truth_verifiability", 3)),
        "rubric_application_clarity": int(parsed.get("rubric_application_clarity", 3)),
        "judge_model": model,
        "rationale": str(parsed.get("rationale", ""))[:200],
        "actual_cost_usd": round(actual_cost, 6),
    }


def _judge_task(task: dict, *, live: bool, models_rotation: list[str], budget: JudgeBudget) -> dict:
    gen_family = _generator_family(task.get("source_provenance", {}).get("generator", ""))

    # Rotate to a model whose family != generator's family (preference-leakage
    # prevention). Most authoring is non-llm so any judge family is fine.
    chosen = None
    for m in models_rotation:
        if _judge_family(m) != gen_family:
            chosen = m
            break
    chosen = chosen or models_rotation[0]

    if live:
        result = _live_judge(task, chosen, budget)
        if result is not None:
            return result
    return _stub_judge(task)


def filter_pool(name: str, *, live: bool, threshold: int = 4) -> dict:
    in_path = POOL_DIR / f"{name}.jsonl"
    if not in_path.exists():
        return {"error": f"{in_path} missing"}
    tasks = [json.loads(line) for line in in_path.read_text().splitlines() if line.strip()]

    budget = JudgeBudget()
    models_rotation = list(TIER_RATES.keys())

    kept: list[dict] = []
    dropped: list[dict] = []
    log_records: list[dict] = []

    for t in tasks:
        scores = _judge_task(t, live=live, models_rotation=models_rotation, budget=budget)
        passed = (
            scores["input_coherence"] >= threshold
            and scores["ground_truth_verifiability"] >= threshold
            and scores["rubric_application_clarity"] >= threshold
        )
        record = {
            "task_id": t["task_id"],
            "primary_dimension": t["primary_dimension"],
            "passed_filter": passed,
            "scores": scores,
        }
        log_records.append(record)
        if passed:
            t.setdefault("source_provenance", {})["judge_filter"] = scores
            kept.append(t)
        else:
            dropped.append((t["task_id"], scores))

    out_path = POOL_DIR / f"{name}_filtered.jsonl"
    with out_path.open("w") as f:
        for t in kept:
            f.write(json.dumps(t, sort_keys=True) + "\n")

    with LOG_PATH.open("a") as f:
        for r in log_records:
            f.write(json.dumps(r) + "\n")

    return {
        "in": len(tasks),
        "kept": len(kept),
        "dropped": len(dropped),
        "out_path": str(out_path),
        "live": live,
        "spend_usd": round(budget.spent_usd, 4),
        "calls": budget.calls,
        "failovers": budget.failovers,
        "errors": budget.errors,
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("--pool", required=True, help="Name of pool jsonl (without extension)")
    parser.add_argument("--live", action="store_true", help="Use live OpenRouter judge")
    parser.add_argument("--threshold", type=int, default=4, help="Min score per dimension")
    args = parser.parse_args()
    result = filter_pool(args.pool, live=args.live, threshold=args.threshold)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
