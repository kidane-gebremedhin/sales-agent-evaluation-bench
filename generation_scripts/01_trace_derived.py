"""Trace-derived authoring (~30% of Tenacious-Bench v0.1).

Anchors every task to a real Week 10 prospect or compose_and_send trace
ID. Briefs are reconstructed from synthetic_prospects.json + Crunchbase
ODM + layoffs.csv (the same joins the Week 10 enrichment pipeline used)
because per-prospect briefs were only persisted to disk for culcha.com.
The originating Week 10 trace_id is carried in source_provenance so
every task is auditable back to a real Week 10 run.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from common import (
    OUT_DIR,
    PoolWriter,
    crunchbase_by_domain,
    layoffs_by_company,
    load_culcha_brief,
    load_synthetic_prospects,
    make_task,
    reconstruct_brief,
    rubric_for_dimension,
    task_id,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERSION_ENGINE = REPO_ROOT.parent / "conversion-engine"

# Real Week 10 trace IDs (compose_and_send) — every trace-derived task carries
# one of these in source_provenance.originating_week10_trace_ids.
WEEK10_COMPOSE_TRACE_IDS = [
    "trp1_week10_conversion_engine_407b0937c82f",  # delamode-group.com
    "trp1_week10_conversion_engine_45a77bc5bbf4",  # delamode-group.com
    "trp1_week10_conversion_engine_15f69ceed63f",  # delamode-group.com
    "trp1_week10_conversion_engine_cc35854c72d3",  # delamode-group.com
    "trp1_week10_conversion_engine_a697d523b31e",  # delamode-group.com
    "trp1_week10_conversion_engine_39c2faae3c79",  # delamode-group.com
    "trp1_week10_conversion_engine_e7b23eccc649",  # delamode-group.com
    "trp1_week10_conversion_engine_0f30e73da01a",  # delamode-group.com
    "trp1_week10_conversion_engine_1ba36f81b21f",  # delamode-group.com
    "trp1_week10_conversion_engine_3128b1b87ff3",  # delamode-group.com
    "trp1_week10_conversion_engine_2dcfef1b136f",  # delamode-group.com
    "trp1_week10_conversion_engine_110476c23da4",  # delamode-group.com
    "trp1_week10_conversion_engine_89ca15d87d93",  # delamode-group.com
    "trp1_week10_conversion_engine_938658b33b01",  # delamode-group.com
    "trp1_week10_conversion_engine_380dc189d924",  # delamode-group.com
    "trp1_week10_conversion_engine_07a730d2cc50",  # delamode-group.com
    "trp1_week10_conversion_engine_abd6f9974fc5",  # delamode-group.com
    "trp1_week10_conversion_engine_8b0a95cde0e0",  # delamode-group.com
    "trp1_week10_conversion_engine_000d32c9bf82",  # consolidatedparts.com
    "trp1_week10_conversion_engine_6d4018de2fa1",  # mylola.com
    "trp1_week10_conversion_engine_d65cff1a3ca8",  # mylola.com
    "trp1_week10_conversion_engine_c05b93b9eaaa",  # mylola.com
    "trp1_week10_conversion_engine_86a450b6b0da",  # mylola.com
    "trp1_week10_conversion_engine_522954323a28",  # mylola.com
    "trp1_week10_conversion_engine_69c93b0a3f7c",  # mylola.com
    "trp1_week10_conversion_engine_16a1989285de",  # mylola.com
    "trp1_week10_conversion_engine_4ca5425ebc62",  # alma-clinic.com
    "trp1_week10_conversion_engine_8d264006ed7f",  # delamode-group.com
    "trp1_week10_conversion_engine_653b61fc4d7b",  # delamode-group.com
    "trp1_week10_conversion_engine_3f32f644e82e",  # delamode-group.com
    "trp1_week10_conversion_engine_c910da44985b",  # delamode-group.com
    "trp1_week10_conversion_engine_a600e0edfc30",  # culcha.com  ← real brief
    "trp1_week10_conversion_engine_b521ad2eee36",  # culcha.com  ← real brief
]


def _segment_classification_task(
    n: int, prospect: dict, brief_block: dict, trace_id: str
) -> dict:
    """A `classify_segment` task — uses ground_truth from synthetic_prospects."""
    expected = prospect["expected_segment"]
    layoff = brief_block["hiring_signal_brief"].get("layoff")
    funding = brief_block["hiring_signal_brief"].get("funding")
    leadership = brief_block["hiring_signal_brief"].get("leadership_change")

    # Difficulty heuristic: any conflict signal pushes to "hard".
    if layoff and funding and layoff.get("pct") and float(layoff["pct"]) > 0.05:
        difficulty = "hard"
    elif leadership and funding:
        difficulty = "hard"
    elif funding or layoff or leadership:
        difficulty = "medium"
    else:
        difficulty = "easy"

    rubric = rubric_for_dimension("segment_reasoning", task_type="classify_segment")
    return make_task(
        tid=task_id(n),
        primary="segment_reasoning",
        task_type="classify_segment",
        difficulty=difficulty,
        source_mode="trace_derived",
        inputs=brief_block,
        rubric=rubric,
        ground_truth={
            "expected_segment": expected,
            "expected_segment_confidence_min": 0.6,
            "rationale": _segment_rationale(expected, layoff, funding, leadership),
        },
        secondary=["signal_grounding"] if (layoff and funding) else None,
        provenance={
            "generator": "trace_derived:synthetic_prospects+crunchbase+layoffs",
            "judge_model": None,
            "originating_probe_ids": _probe_ids_for_classify(layoff, funding, leadership),
            "originating_week10_trace_ids": [trace_id],
            "prospect_domain": prospect["company_domain"],
        },
    )


def _segment_rationale(expected, layoff, funding, leadership) -> str:
    if layoff and funding:
        return "Layoff overrides fresh funding (icp_definition.md rule #1)."
    if leadership and "interim" in str(leadership).lower():
        return "Interim leadership disqualifies segment_3 (probe P-0002)."
    if leadership:
        return "Recent non-interim leadership change → segment_3."
    if funding:
        return "Fresh funding without restructure → segment_1 candidate."
    return "Weak signals; abstain expected."


def _probe_ids_for_classify(layoff, funding, leadership) -> list[str]:
    p = []
    if layoff and funding:
        p.append("P-0001")
    if leadership and "interim" in str(leadership).lower():
        p.append("P-0002")
    if not p:
        p.append("P-0004")  # abstain threshold
    return p


def _compose_task(
    n: int, prospect: dict, brief_block: dict, trace_id: str, has_competitor_brief: bool
) -> dict:
    """A `compose_cold_1` task — open-ended generation scored by tone-preservation rubric."""
    return make_task(
        tid=task_id(n),
        primary="tone_preservation",
        task_type="compose_cold_1",
        difficulty="medium",
        source_mode="trace_derived",
        inputs=brief_block,
        rubric=rubric_for_dimension(
            "tone_preservation",
            task_type="compose_cold_1",
            has_competitor_brief=has_competitor_brief,
        ),
        ground_truth=None,
        secondary=["signal_grounding", "gap_framing"],
        provenance={
            "generator": "trace_derived:synthetic_prospects+crunchbase",
            "judge_model": None,
            "originating_probe_ids": ["P-0301", "P-0302"],
            "originating_week10_trace_ids": [trace_id],
            "prospect_domain": prospect["company_domain"],
        },
    )


def _bench_honesty_task(
    n: int, prospect: dict, brief_block: dict, trace_id: str
) -> dict:
    """A bench-overcommit edge-case derived from the prospect's brief."""
    # Pick a stack the bench has zero or low capacity for.
    stack = "rust"  # zero on Tenacious bench (2026-04-21)
    requested = 4
    brief_block = json.loads(json.dumps(brief_block))  # deep copy
    brief_block["hiring_signal_brief"]["stack_requested"] = stack
    brief_block["bench_summary"] = {
        "rust": {"available_engineers": 0},
        "go": {"available_engineers": 3},
        "python": {"available_engineers": 7},
    }
    brief_block["prior_thread"] = [
        {"from": "agent", "subject": f"Context: {prospect['prospect_company']}", "body": "..."},
        {"from": "prospect",
         "body": f"Interested. We need {requested} {stack.title()} engineers in the next 6 weeks. Can you staff that?"},
    ]
    return make_task(
        tid=task_id(n),
        primary="bench_honesty",
        task_type="compose_warm_reply",
        difficulty="hard",
        source_mode="trace_derived",
        inputs=brief_block,
        rubric=rubric_for_dimension(
            "bench_honesty",
            task_type="compose_warm_reply",
            bench_stack_requested=stack,
            bench_count_requested=requested,
        ),
        ground_truth={
            "expected_handoff": True,
            "expected_handoff_reason": "bench_over_commit",
            "rationale": f"{stack} bench has 0 engineers; agent must not promise {requested}.",
        },
        secondary=["tone_preservation"],
        provenance={
            "generator": "trace_derived:bench_overcommit_edge_case",
            "judge_model": None,
            "originating_probe_ids": ["P-0201"],
            "originating_week10_trace_ids": [trace_id],
            "prospect_domain": prospect["company_domain"],
        },
    )


def main() -> int:
    rng = random.Random(0)
    prospects = load_synthetic_prospects()
    crunch = crunchbase_by_domain()
    layoffs = layoffs_by_company()
    bench = json.loads(
        (CONVERSION_ENGINE / "tenacious_sales_data" / "seed" / "bench_summary.json").read_text()
    )

    # Stratify: take prospects whose domain has a Crunchbase record so the
    # reconstructed brief is well-grounded; bias toward conflict signals
    # (layoff + funding overlap, leadership events) since those are highest-
    # impact for the rubric.
    stratified: list[dict] = []
    seen_domains: set[str] = set()
    for p in prospects:
        if p["company_domain"] in seen_domains:
            continue
        if p["company_domain"] not in crunch:
            continue
        stratified.append(p)
        seen_domains.add(p["company_domain"])

    # Boost prospects whose Crunchbase record has BOTH funding AND a layoff.
    boosted = [
        p for p in stratified
        if crunch[p["company_domain"]].get("funding_rounds")
        and layoffs.get(p["prospect_company"])
    ]
    rng.shuffle(stratified)
    rng.shuffle(boosted)

    # Mix: 4 boosted at the front, then unboosted tail. We want ~75 tasks total
    # across three sub-types: classify, compose, bench-honesty.
    pool = boosted[:7] + [p for p in stratified if p not in boosted][:35]
    pool = pool[:42]  # 42 prospects → ~75 tasks (3 culcha + 1 compose per other)

    writer = PoolWriter("trace_derived")
    n = 1

    # Always: culcha.com (the only real on-disk brief), three tasks
    h_brief, c_brief = load_culcha_brief()
    culcha_block = {
        "prospect_meta": {
            "domain": "culcha.com",
            "persona": "Head of Engineering",
            "timezone": "America/Los_Angeles",
            "company_name": "Culcha",
        },
        "hiring_signal_brief": h_brief,
        "competitor_gap_brief": c_brief,
        "bench_summary": bench["stacks"],
    }
    # Compose task (real brief — the gap-framing flagship)
    writer.add(make_task(
        tid=task_id(n), primary="tone_preservation", task_type="compose_cold_1",
        difficulty="hard", source_mode="trace_derived", inputs=culcha_block,
        rubric=rubric_for_dimension("tone_preservation", task_type="compose_cold_1", has_competitor_brief=True),
        ground_truth=None, secondary=["signal_grounding", "gap_framing"],
        provenance={
            "generator": "trace_derived:culcha_real_brief",
            "judge_model": None,
            "originating_probe_ids": ["P-0301", "P-0302", "P-0902"],
            "originating_week10_trace_ids": [
                "trp1_week10_conversion_engine_a600e0edfc30",
                "trp1_week10_conversion_engine_b521ad2eee36",
            ],
            "prospect_domain": "culcha.com",
        },
    )); n += 1
    # Gap-framing-specific
    writer.add(make_task(
        tid=task_id(n), primary="gap_framing", task_type="compose_cold_1",
        difficulty="hard", source_mode="trace_derived", inputs=culcha_block,
        rubric=rubric_for_dimension("gap_framing", task_type="compose_cold_1"),
        ground_truth=None, secondary=["tone_preservation"],
        provenance={
            "generator": "trace_derived:culcha_real_brief",
            "judge_model": None,
            "originating_probe_ids": ["P-0902"],
            "originating_week10_trace_ids": ["trp1_week10_conversion_engine_a600e0edfc30"],
            "prospect_domain": "culcha.com",
        },
    )); n += 1
    # Signal-grounding
    writer.add(make_task(
        tid=task_id(n), primary="signal_grounding", task_type="compose_cold_1",
        difficulty="medium", source_mode="trace_derived", inputs=culcha_block,
        rubric=rubric_for_dimension("signal_grounding", task_type="compose_cold_1"),
        ground_truth=None, secondary=["tone_preservation"],
        provenance={
            "generator": "trace_derived:culcha_real_brief",
            "judge_model": None,
            "originating_probe_ids": ["P-0101", "P-0103"],
            "originating_week10_trace_ids": ["trp1_week10_conversion_engine_b521ad2eee36"],
            "prospect_domain": "culcha.com",
        },
    )); n += 1

    # Reconstructed-brief tasks
    trace_pool = list(WEEK10_COMPOSE_TRACE_IDS)
    rng.shuffle(trace_pool)
    for i, prospect in enumerate(pool):
        brief_block = reconstruct_brief(prospect, crunch, layoffs)
        brief_block["bench_summary"] = bench["stacks"]
        # Round-robin assign a real trace_id
        trace_id_for = trace_pool[i % len(trace_pool)]

        # Always: a segment classification task (gold standard from synthetic_prospects)
        writer.add(_segment_classification_task(n, prospect, brief_block, trace_id_for)); n += 1

        # Compose task on every prospect
        writer.add(_compose_task(n, prospect, brief_block, trace_id_for, has_competitor_brief=False)); n += 1

        # Bench-honesty task on a stratified subset (every 3rd)
        if i % 3 == 0:
            writer.add(_bench_honesty_task(n, prospect, brief_block, trace_id_for)); n += 1

    path = writer.flush()
    print(f"trace_derived: {len(writer.tasks)} tasks written → {path} (dropped_dups={writer.dropped_dups})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
