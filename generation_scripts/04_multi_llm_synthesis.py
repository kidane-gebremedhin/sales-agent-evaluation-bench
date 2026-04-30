"""Multi-LLM synthesis (~25% of Tenacious-Bench v0.1).

Magpie-style self-instruction (Xu et al., 2024) adapted for Tenacious
B2B sales tasks. A single dev-tier LLM (Qwen3-Next or DeepSeek V3.2)
generates hard variations of each authoring seed; outputs are
deduplicated, structurally validated, and pass through judge_filter
in a separate pass.

Preference-leakage policy: the model that generates a task is recorded
in source_provenance.generator. The judge filter (judge_filter.py)
will refuse to use a same-family judge on that task — see
methodology.md for the rotation table.

Cost discipline: hard cap of $0.20 across the entire synthesis run,
estimated by approx-token billing. The model is rotated across calls
to avoid lock-in. If the cap is hit, the script halts cleanly and
records what was generated.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from common import (
    PoolWriter,
    load_bench_summary,
    make_task,
    rubric_for_dimension,
    task_id,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = REPO_ROOT / "generation_scripts" / "synthesis_log.jsonl"
COST_CAP_USD = float(os.environ.get("TB_SYNTH_COST_CAP_USD", "0.20"))


# Rotated dev-tier models — rotation policy committed in methodology.md.
SYNTH_MODELS = [
    "deepseek/deepseek-v3.2",
    "qwen/qwen3.5-4b-instruct",
]
TIER_RATES = {
    "deepseek/deepseek-v3.2": (0.20, 0.80),
    "qwen/qwen3.5-4b-instruct": (0.10, 0.30),
}


def _approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _est_cost(model: str, in_t: int, out_t: int) -> float:
    rin, rout = TIER_RATES.get(model, (0.20, 0.80))
    return (in_t * rin + out_t * rout) / 1_000_000


@dataclass
class SynthBudget:
    cap_usd: float = COST_CAP_USD
    spent_usd: float = 0.0
    calls: int = 0
    halted: bool = False


def _call_llm(prompt: str, model: str, budget: SynthBudget, max_out: int = 600) -> dict | None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None
    in_t = _approx_tokens(prompt)
    est = _est_cost(model, in_t, max_out)
    if (budget.spent_usd + est) > budget.cap_usd:
        budget.halted = True
        return None

    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_out,
        "temperature": 0.7,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")
    base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    req = urllib.request.Request(
        base + "/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/tenacious-bench",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError):
        return None
    usage = data.get("usage") or {}
    actual_in = usage.get("prompt_tokens", in_t)
    actual_out = usage.get("completion_tokens", max_out)
    actual_cost = usage.get("cost") or _est_cost(model, actual_in, actual_out)
    budget.spent_usd += actual_cost
    budget.calls += 1
    text = data["choices"][0]["message"]["content"]
    return {"text": text, "model": model, "cost": actual_cost,
            "in_tokens": actual_in, "out_tokens": actual_out}


# Seed prompts — anchored in the Week 10 failure taxonomy. Each seed targets
# a Tenacious failure mode; the LLM is asked to produce ONE variation that
# exercises the same mode under a new prospect setup.

SEEDS = [
    {
        "primary": "segment_reasoning",
        "task_type": "classify_segment",
        "probes": ["P-0001"],
        "instruction": (
            "Generate a hiring_signal_brief for a synthetic B2B prospect that EXERCISES the "
            "Tenacious failure mode where a fresh funding event AND a recent layoff overlap. "
            "The correct segment must be 'segment_2_mid_market_restructure' because layoff overrides funding. "
            "Vary: layoff_pct (must be 0.05–0.30), funding_round, layoff scope, prospect_meta.persona, headcount_band. "
            "Make it tricky but the rule must still apply."
        ),
        "n": 12,
    },
    {
        "primary": "segment_reasoning",
        "task_type": "classify_segment",
        "probes": ["P-0002"],
        "instruction": (
            "Generate a brief where a recent leadership change is INTERIM or ACTING — the "
            "expected_segment must be 'abstain' (interim disqualifies segment_3). Vary the "
            "title phrasing, days_ago, and persona."
        ),
        "n": 8,
    },
    {
        "primary": "tone_preservation",
        "task_type": "compose_cold_1",
        "probes": ["P-0301", "P-0302"],
        "instruction": (
            "Generate a hiring_signal_brief for a Segment 1 prospect with strong public funding signal, "
            "a Python/data hiring need, and an ai_maturity_score of 1 or 2. Include all fields the brief "
            "schema expects (funding with source_url, open_roles_today, persona, timezone, headcount_band). "
            "Goal: the resulting compose_cold_1 task must require a tone-preserving direct, grounded body "
            "of ≤120 words."
        ),
        "n": 12,
    },
    {
        "primary": "bench_honesty",
        "task_type": "compose_warm_reply",
        "probes": ["P-0201", "P-0202"],
        "instruction": (
            "Generate a prior_thread + bench_summary scenario where the prospect demands MORE engineers "
            "in a specific stack (rust, ml, fullstack_nestjs, infra) than the bench shows. Specify the "
            "stack and requested_count clearly. Make sure the bench_summary explicitly shows fewer than "
            "requested in that stack."
        ),
        "n": 10,
    },
    {
        "primary": "gap_framing",
        "task_type": "compose_cold_1",
        "probes": ["P-0902"],
        "instruction": (
            "Generate a Segment 4 hiring brief and a competitor_gap_brief with 3-5 peer_evidence entries "
            "(each with source_url). The prospect's ai_maturity_score must be 2 or 3. The task tests "
            "whether the agent frames the gap as research not as 'you're behind'."
        ),
        "n": 10,
    },
    {
        "primary": "signal_grounding",
        "task_type": "compose_cold_1",
        "probes": ["P-0101", "P-0103"],
        "instruction": (
            "Generate a hiring_signal_brief whose open_roles_today is between 2 and 4 and whose "
            "open_roles_60_days_ago is similar — i.e. WEAK velocity. Funding must be present (with or "
            "without source_url). The task asks the agent NOT to over-claim 'aggressive hiring'."
        ),
        "n": 8,
    },
]


def _build_prompt(seed: dict) -> str:
    return (
        f"You are authoring a single Tenacious-Bench task. Tenacious is a B2B engineering-talent "
        f"company; the agent we are evaluating writes outbound sales messages.\n\n"
        f"Failure mode to exercise: {seed['probes']}\n"
        f"Primary rubric dimension: {seed['primary']}\n"
        f"Task type: {seed['task_type']}\n\n"
        f"Instruction: {seed['instruction']}\n\n"
        f"Return STRICT JSON with these top-level keys:\n"
        f"  prospect_meta: {{domain, persona, timezone, company_name}}\n"
        f"  hiring_signal_brief: {{... see instruction ...}}\n"
        f"  competitor_gap_brief: object or null\n"
        f"  prior_thread: array (only for compose_warm_reply / compose_warm_objection)\n"
        f"  expected_label: required for segment_reasoning (one of: segment_1_series_a_b, "
        f"segment_2_mid_market_restructure, segment_3_leadership_transition, "
        f"segment_4_specialized_capability, abstain)\n"
        f"  rationale: one sentence explaining why this exercises the failure mode\n\n"
        f"Use realistic but synthetic domains (synth-ml-{{n}}.example), realistic role counts, "
        f"and proper ISO timezone names. Do not include any real company names."
    )


def _validate_and_wrap(payload: dict, n: int, seed: dict, model: str, cost: float, bench_stacks: dict) -> dict | None:
    """Convert LLM JSON to a Tenacious-Bench task. Returns None if invalid."""
    if not isinstance(payload, dict):
        return None
    pm = payload.get("prospect_meta") or {}
    hsb = payload.get("hiring_signal_brief") or {}
    if not pm.get("domain") or not hsb:
        return None

    bench_block = {k: {"available_engineers": v.get("available_engineers", 0),
                       "skill_subsets": v.get("skill_subsets", [])}
                   for k, v in bench_stacks.items() if isinstance(v, dict)}
    inputs = {
        "prospect_meta": pm,
        "hiring_signal_brief": hsb,
        "competitor_gap_brief": payload.get("competitor_gap_brief"),
        "prior_thread": payload.get("prior_thread"),
        "bench_summary": bench_block,
    }

    primary = seed["primary"]
    extra = {}
    if primary == "bench_honesty":
        # Try to extract stack/requested from prior_thread or fallback
        thread_text = json.dumps(inputs.get("prior_thread") or [])
        m = re.search(r"\b(\d+)\s+(rust|go|python|ml|infra|fullstack[_ ]nestjs|frontend|data)\b", thread_text, re.IGNORECASE)
        if m:
            extra["bench_count_requested"] = int(m.group(1))
            extra["bench_stack_requested"] = m.group(2).lower().replace(" ", "_")
        else:
            extra["bench_count_requested"] = 4
            extra["bench_stack_requested"] = "rust"

    rubric = rubric_for_dimension(
        primary,
        task_type=seed["task_type"],
        has_competitor_brief=bool(inputs.get("competitor_gap_brief")),
        **extra,
    )
    gt = None
    if primary == "segment_reasoning":
        expected = payload.get("expected_label") or "abstain"
        gt = {
            "expected_segment": expected,
            "expected_segment_confidence_min": 0.6 if expected != "abstain" else 0.4,
            "rationale": payload.get("rationale", ""),
        }
    elif primary == "bench_honesty":
        bench_avail = bench_block.get(extra["bench_stack_requested"], {}).get("available_engineers", 0)
        gt = {
            "expected_handoff": bench_avail < extra["bench_count_requested"],
            "expected_handoff_reason": "bench_over_commit" if bench_avail < extra["bench_count_requested"] else None,
            "rationale": payload.get("rationale", ""),
        }

    return make_task(
        tid=task_id(n),
        primary=primary,
        task_type=seed["task_type"],
        difficulty="hard",
        source_mode="multi_llm_synthesis",
        inputs=inputs,
        rubric=rubric,
        ground_truth=gt,
        secondary=None,
        provenance={
            "generator": f"multi_llm_synthesis:{model}",
            "judge_model": None,  # filled in by judge_filter.py
            "originating_probe_ids": seed["probes"],
            "originating_week10_trace_ids": ["trp1_week10_conversion_engine_a600e0edfc30"],
            "synth_seed_instruction": seed["instruction"][:120] + "...",
            "synth_cost_usd": round(cost, 6),
        },
    )


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    try:
        return json.loads(text.strip().strip("`"))
    except json.JSONDecodeError:
        pass
    # Try to peel a code fence
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def main() -> int:
    bench = load_bench_summary()
    bench_stacks = bench["stacks"]
    writer = PoolWriter("multi_llm_synthesis")
    budget = SynthBudget()
    n = 3001
    log_records: list[dict] = []

    for seed in SEEDS:
        prompt = _build_prompt(seed)
        for i in range(seed["n"]):
            if budget.halted:
                break
            model = SYNTH_MODELS[i % len(SYNTH_MODELS)]
            resp = _call_llm(prompt, model, budget)
            if resp is None:
                if budget.halted:
                    print(f"  cap reached at ${budget.spent_usd:.4f}; halting cleanly")
                    break
                log_records.append({"seed": seed["probes"], "i": i, "status": "no_response"})
                continue
            payload = _extract_json(resp["text"])
            if payload is None:
                log_records.append({"seed": seed["probes"], "i": i, "status": "json_parse_failed", "preview": resp["text"][:200]})
                continue
            task = _validate_and_wrap(payload, n, seed, resp["model"], resp["cost"], bench_stacks)
            if task is None:
                log_records.append({"seed": seed["probes"], "i": i, "status": "validation_failed"})
                continue
            if writer.add(task):
                n += 1
                log_records.append({"seed": seed["probes"], "i": i, "status": "ok",
                                    "task_id": task["task_id"], "model": resp["model"], "cost_usd": resp["cost"]})
            else:
                log_records.append({"seed": seed["probes"], "i": i, "status": "duplicate"})
        if budget.halted:
            break

    path = writer.flush()
    with LOG_PATH.open("a") as f:
        for r in log_records:
            f.write(json.dumps(r) + "\n")

    print(json.dumps({
        "tasks_generated": len(writer.tasks),
        "duplicates_dropped": writer.dropped_dups,
        "spend_usd": round(budget.spent_usd, 5),
        "calls": budget.calls,
        "cap_hit": budget.halted,
        "out_path": str(path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
