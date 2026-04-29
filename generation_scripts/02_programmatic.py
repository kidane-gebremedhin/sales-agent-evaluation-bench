"""Programmatic authoring (~30% of Tenacious-Bench v0.1).

Templates with structured slots, populated by combinatorial sweeps:
- segment classification × layoff_pct × funding_round × leadership
- bench overcommit × stack × requested_count
- AI-maturity gate × ai_maturity_score × persona
- signal-grounding × open_roles_today × source_url_present

Each template expands to 5–25 tasks depending on slot cardinality.
"""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

from common import (
    PoolWriter,
    load_bench_summary,
    make_task,
    rubric_for_dimension,
    task_id,
)


def _classify_layoff_funding_sweep(start_n: int, writer: PoolWriter) -> int:
    """P-0001 — layoff+funding misclassification, swept across pct & funding stage."""
    n = start_n
    layoff_pcts = [0.05, 0.08, 0.12, 0.18, 0.25]
    funding_stages = ["seed", "series_a", "series_b", "series_c"]
    funding_amounts = [3_000_000, 9_000_000, 14_000_000, 35_000_000]
    for pct, (stage, amt) in itertools.product(
        layoff_pcts, zip(funding_stages, funding_amounts)
    ):
        domain = f"prospect-{n:04d}.example"
        inputs = {
            "prospect_meta": {
                "domain": domain,
                "persona": "VP Engineering",
                "timezone": "America/New_York",
                "company_name": f"Synth-{n:04d}",
            },
            "hiring_signal_brief": {
                "funding": {
                    "round": stage,
                    "amount_usd": amt,
                    "closed_at": "2026-02-08",
                    "source_url": f"https://example/funding-{n}",
                },
                "layoff": {
                    "pct": pct,
                    "occurred_at": "2026-03-20",
                    "scope": "engineering",
                    "source_url": f"https://layoffs.fyi/example-{n}",
                },
                "leadership_change": None,
                "open_roles_today": 4,
                "open_roles_60_days_ago": 6,
                "ai_maturity_score": 1,
            },
            "competitor_gap_brief": None,
            "bench_summary": {"as_of": "2026-04-21", "total_engineers_on_bench": 36},
        }
        # Rule: layoff (any pct >= 0.05) overrides funding → segment_2.
        # If pct < 0.05 the layoff is too small to override → segment_1.
        if pct >= 0.05:
            expected = "segment_2_mid_market_restructure"
            difficulty = "hard" if pct in (0.05, 0.08) else "medium"
        else:
            expected = "segment_1_series_a_b"
            difficulty = "medium"

        writer.add(make_task(
            tid=task_id(n),
            primary="segment_reasoning",
            task_type="classify_segment",
            difficulty=difficulty,
            source_mode="programmatic",
            inputs=inputs,
            rubric=rubric_for_dimension("segment_reasoning", task_type="classify_segment"),
            ground_truth={
                "expected_segment": expected,
                "expected_segment_confidence_min": 0.6,
                "rationale": "Layoff override rule (P-0001) — any pct ≥ 0.05 within 120 days overrides fresh funding."
            },
            provenance={
                "generator": "programmatic:layoff_funding_sweep",
                "judge_model": None,
                "originating_probe_ids": ["P-0001"],
                "originating_week10_trace_ids": ["trp1_week10_conversion_engine_b3f5aa034a4f"],
                "sweep_params": {"layoff_pct": pct, "funding_stage": stage, "amount_usd": amt},
            },
        ))
        n += 1
    return n


def _classify_interim_cto_sweep(start_n: int, writer: PoolWriter) -> int:
    """P-0002 — interim CTO disqualifies segment_3."""
    n = start_n
    titles = [
        "Interim CTO",
        "Interim VP Engineering",
        "Acting CTO",
        "Chief Technology Officer (Interim)",
        "Chief Technology Officer",       # NOT interim → should classify segment_3
        "VP Engineering",                  # NOT interim → segment_3 candidate
    ]
    days_ago = [15, 45, 90]
    for title, dago in itertools.product(titles, days_ago):
        domain = f"prospect-{n:04d}.example"
        is_interim = "interim" in title.lower() or "acting" in title.lower()
        inputs = {
            "prospect_meta": {"domain": domain, "persona": "CTO", "timezone": "America/Los_Angeles"},
            "hiring_signal_brief": {
                "leadership_change": {"title": title, "started_days_ago": dago, "source_url": f"https://press/{n}"},
                "funding": None,
                "layoff": None,
                "open_roles_today": 6,
                "ai_maturity_score": 2,
            },
            "competitor_gap_brief": None,
            "bench_summary": {"as_of": "2026-04-21"},
        }
        if is_interim:
            expected = "abstain"
            difficulty = "hard" if dago <= 30 else "medium"
        else:
            expected = "segment_3_leadership_transition"
            difficulty = "easy"
        writer.add(make_task(
            tid=task_id(n),
            primary="segment_reasoning",
            task_type="classify_segment",
            difficulty=difficulty,
            source_mode="programmatic",
            inputs=inputs,
            rubric=rubric_for_dimension("segment_reasoning", task_type="classify_segment"),
            ground_truth={
                "expected_segment": expected,
                "expected_segment_confidence_min": 0.6 if not is_interim else 0.4,
                "rationale": "Interim/Acting disqualifier (P-0002)" if is_interim else "Substantive leadership transition → segment_3.",
            },
            provenance={
                "generator": "programmatic:interim_cto_sweep",
                "judge_model": None,
                "originating_probe_ids": ["P-0002"],
                "originating_week10_trace_ids": ["trp1_week10_conversion_engine_b3f5aa034a4f"],
                "sweep_params": {"title": title, "started_days_ago": dago},
            },
        ))
        n += 1
    return n


def _bench_overcommit_sweep(start_n: int, writer: PoolWriter, bench: dict) -> int:
    """P-0201 — bench overcommit across stack × requested_count."""
    n = start_n
    stacks_with_capacity = {
        "rust": 0, "go": 3, "python": 7, "ml": 5, "infra": 4,
        "fullstack_nestjs": 2, "frontend": 6, "data": 9,
    }
    requested_counts = [1, 2, 3, 4, 6, 10]
    for stack, requested in itertools.product(stacks_with_capacity.keys(), requested_counts):
        available = stacks_with_capacity[stack]
        # Only generate cases where overcommit is a real risk OR a clean control.
        if requested - available > 6 or (available > 0 and requested <= available - 2):
            # too pathological or too easy — skip to keep mass at the boundary
            continue
        domain = f"prospect-{n:04d}.example"
        inputs = {
            "prospect_meta": {"domain": domain, "persona": "VP Engineering", "timezone": "America/New_York"},
            "hiring_signal_brief": {
                "primary_segment_match": "segment_1_series_a_b",
                "open_roles_today": requested,
                "stack_requested": stack,
            },
            "prior_thread": [
                {"from": "agent", "subject": f"Context: your team", "body": "..."},
                {"from": "prospect",
                 "body": f"We need {requested} {stack.replace('_',' ').title()} engineers in the next 6 weeks. Can you staff that?"},
            ],
            "bench_summary": {k: {"available_engineers": v} for k, v in stacks_with_capacity.items()},
        }
        difficulty = "adversarial" if available == 0 and requested >= 3 else ("hard" if available < requested else "easy")
        expected_handoff = available < requested
        writer.add(make_task(
            tid=task_id(n),
            primary="bench_honesty",
            task_type="compose_warm_reply",
            difficulty=difficulty,
            source_mode="programmatic",
            inputs=inputs,
            rubric=rubric_for_dimension(
                "bench_honesty",
                task_type="compose_warm_reply",
                bench_stack_requested=stack,
                bench_count_requested=requested,
            ),
            ground_truth={
                "expected_handoff": expected_handoff,
                "expected_handoff_reason": "bench_over_commit" if expected_handoff else None,
                "rationale": f"{stack} bench has {available}; prospect requested {requested}.",
            },
            secondary=["tone_preservation"],
            provenance={
                "generator": "programmatic:bench_overcommit_sweep",
                "judge_model": None,
                "originating_probe_ids": ["P-0201", "P-0202", "P-0203"],
                "originating_week10_trace_ids": ["trp1_week10_conversion_engine_407b0937c82f"],
                "sweep_params": {"stack": stack, "requested": requested, "available": available},
            },
        ))
        n += 1
    return n


def _ai_maturity_gate_sweep(start_n: int, writer: PoolWriter) -> int:
    """P-0003 — Segment 4 AI-maturity hard gate sweep across maturity × persona."""
    n = start_n
    maturity_scores = [0, 1, 2, 3]
    personas = ["VP Engineering", "Head of Data", "Director of ML", "CTO"]
    for score, persona in itertools.product(maturity_scores, personas):
        domain = f"prospect-{n:04d}.example"
        inputs = {
            "prospect_meta": {"domain": domain, "persona": persona, "timezone": "Africa/Addis_Ababa"},
            "hiring_signal_brief": {
                "ai_maturity_score": score,
                "named_ai_ml_leadership": score >= 2,
                "open_roles_today": 5,
                "ai_adjacent_open_roles_pct": 0.10 * score,
            },
            "competitor_gap_brief": {
                "peer_evidence": [
                    {"peer": "PeerCo A", "practice": "AI platform engineering team",
                     "source_url": "https://example/peer-a"},
                ],
            },
            "bench_summary": {"ml": {"available_engineers": 5}},
        }
        # Segment 4 requires ai_maturity >= 2; below that the composer should NOT
        # try a Segment 4 pitch.
        if score >= 2:
            expected = "segment_4_specialized_capability"
            difficulty = "medium"
        else:
            expected = "abstain"
            difficulty = "hard"
        writer.add(make_task(
            tid=task_id(n),
            primary="segment_reasoning",
            task_type="classify_segment",
            difficulty=difficulty,
            source_mode="programmatic",
            inputs=inputs,
            rubric=rubric_for_dimension("segment_reasoning", task_type="classify_segment"),
            ground_truth={
                "expected_segment": expected,
                "expected_segment_confidence_min": 0.6 if score >= 2 else 0.4,
                "rationale": "AI-maturity ≥ 2 required for segment_4 (P-0003).",
            },
            provenance={
                "generator": "programmatic:ai_maturity_gate_sweep",
                "judge_model": None,
                "originating_probe_ids": ["P-0003"],
                "originating_week10_trace_ids": ["trp1_week10_conversion_engine_a600e0edfc30"],
                "sweep_params": {"ai_maturity_score": score, "persona": persona},
            },
        ))
        n += 1
    return n


def _open_roles_velocity_sweep(start_n: int, writer: PoolWriter) -> int:
    """P-0101 — over-claiming aggressive hiring at low velocity."""
    n = start_n
    today_pairs = [
        # (today, 60d_ago) → low velocity vs high velocity
        (3, 2), (4, 3), (5, 5),                   # low / flat
        (8, 4), (12, 6), (20, 8),                # high
    ]
    for today, prior in today_pairs:
        for has_funding in (True, False):
            domain = f"prospect-{n:04d}.example"
            inputs = {
                "prospect_meta": {"domain": domain, "persona": "VP Engineering"},
                "hiring_signal_brief": {
                    "open_roles_today": today,
                    "open_roles_60_days_ago": prior,
                    "funding": (
                        {"round": "series_b", "amount_usd": 14_000_000,
                         "closed_at": "2026-02-08", "source_url": "https://example/f"}
                        if has_funding else None
                    ),
                    "ai_maturity_score": 1,
                },
                "competitor_gap_brief": None,
                "bench_summary": {"python": {"available_engineers": 7}},
            }
            difficulty = "hard" if today < 5 else "medium"
            writer.add(make_task(
                tid=task_id(n),
                primary="signal_grounding",
                task_type="compose_cold_1",
                difficulty=difficulty,
                source_mode="programmatic",
                inputs=inputs,
                rubric=rubric_for_dimension("signal_grounding", task_type="compose_cold_1"),
                ground_truth=None,
                secondary=["tone_preservation"],
                provenance={
                    "generator": "programmatic:open_roles_velocity_sweep",
                    "judge_model": None,
                    "originating_probe_ids": ["P-0101", "P-0103"],
                    "originating_week10_trace_ids": ["trp1_week10_conversion_engine_b521ad2eee36"],
                    "sweep_params": {"today": today, "prior": prior, "has_funding": has_funding},
                },
            ))
            n += 1
    return n


def main() -> int:
    bench = load_bench_summary()
    writer = PoolWriter("programmatic")
    n = 1001  # programmatic IDs start at 1001 to keep ID ranges per-mode tidy
    n = _classify_layoff_funding_sweep(n, writer)
    n = _classify_interim_cto_sweep(n, writer)
    n = _bench_overcommit_sweep(n, writer, bench)
    n = _ai_maturity_gate_sweep(n, writer)
    n = _open_roles_velocity_sweep(n, writer)
    path = writer.flush()
    print(f"programmatic: {len(writer.tasks)} tasks written → {path} (dropped_dups={writer.dropped_dups})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
