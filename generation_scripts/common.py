"""Common helpers for Tenacious-Bench task authoring (Act II).

Every authoring mode (trace-derived, programmatic, multi-LLM synthesis,
hand-authored) imports from here so task-id minting, dedup, schema
validation, and metadata stamping stay consistent."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# Repo paths
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "tenacious_bench_v0.1" / "_pool"  # pre-partition pool
SCHEMA_PATH = REPO_ROOT / "schema.json"
CONVERSION_ENGINE = REPO_ROOT.parent / "conversion-engine"
WEEK10_DATA = CONVERSION_ENGINE / "data"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# IDs & hashes
# --------------------------------------------------------------------------


def task_id(n: int) -> str:
    return f"TB-{n:04d}"


def task_hash(task: dict) -> str:
    """Stable content hash for dedup.

    Hashes the task's input + ground_truth + primary_dimension. Two tasks
    with the same content but different IDs collapse to the same hash."""
    payload = {
        "primary_dimension": task.get("primary_dimension"),
        "task_type": task.get("task_type"),
        "input": task.get("input"),
        "ground_truth": task.get("ground_truth"),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()[:16]


# --------------------------------------------------------------------------
# Reference data loaders (Week 10 corpus)
# --------------------------------------------------------------------------


def load_synthetic_prospects() -> list[dict]:
    return json.loads((WEEK10_DATA / "synthetic_prospects.json").read_text())


def load_crunchbase() -> list[dict]:
    return json.loads((WEEK10_DATA / "crunchbase_odm_sample.json").read_text())


def load_layoffs() -> list[dict]:
    with (WEEK10_DATA / "layoffs.csv").open() as f:
        return list(csv.DictReader(f))


def load_bench_summary() -> dict:
    return json.loads(
        (CONVERSION_ENGINE / "tenacious_sales_data" / "seed" / "bench_summary.json").read_text()
    )


def load_culcha_brief() -> tuple[dict, dict]:
    """Returns (hiring_signal_brief, competitor_gap_brief) for culcha.com —
    the only fully persisted real brief in the Week 10 repo."""
    base = CONVERSION_ENGINE / "eval" / "briefs" / "culcha.com"
    h = json.loads((base / "hiring_signal_brief.json").read_text())
    c = json.loads((base / "competitor_gap_brief.json").read_text())
    return h, c


# --------------------------------------------------------------------------
# Brief reconstruction (for trace-derived tasks where the brief was not persisted)
# --------------------------------------------------------------------------


def crunchbase_by_domain() -> dict[str, dict]:
    return {c["domain"]: c for c in load_crunchbase() if c.get("domain")}


def layoffs_by_company() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in load_layoffs():
        out.setdefault(row["Company"], []).append(row)
    return out


def reconstruct_brief(prospect: dict, crunch_idx: dict, layoff_idx: dict) -> dict:
    """Reconstruct a hiring_signal_brief shape from public sources, the way the
    Week 10 enrichment pipeline would have done it. Used for trace-derived
    tasks whose original brief was not persisted to disk."""
    domain = prospect["company_domain"]
    crunch = crunch_idx.get(domain, {})
    layoffs = layoff_idx.get(prospect.get("prospect_company", ""), [])

    funding_rounds = sorted(
        (r for r in crunch.get("funding_rounds", []) if r.get("closed_at")),
        key=lambda r: r.get("closed_at"),
        reverse=True,
    )
    latest_funding = funding_rounds[0] if funding_rounds else None

    layoff_rec = sorted(layoffs, key=lambda r: r.get("Date", ""), reverse=True)
    latest_layoff = layoff_rec[0] if layoff_rec else None

    leadership_events = crunch.get("leadership_hire_events") or []
    latest_leadership = leadership_events[-1] if leadership_events else None

    # Synthetic but bounded values; used only for grounding the rubric.
    open_roles_today = (hash(domain) % 18)  # 0..17
    open_roles_60d_ago = max(0, open_roles_today - 2 + (hash(domain + "x") % 5))

    return {
        "prospect_meta": {
            "domain": domain,
            "persona": prospect.get("prospect_title"),
            "timezone": prospect.get("prospect_timezone"),
            "company_name": prospect.get("prospect_company"),
        },
        "hiring_signal_brief": {
            "primary_segment_match": prospect.get("expected_segment"),
            "ai_maturity_score": prospect.get("expected_ai_maturity_score"),
            "headcount_band": crunch.get("headcount_band"),
            "headcount_point": crunch.get("headcount_point"),
            "funding": (
                {
                    "round": latest_funding.get("stage"),
                    "amount_usd": latest_funding.get("amount_usd"),
                    "closed_at": latest_funding.get("closed_at"),
                    "source_url": latest_funding.get("source_url"),
                }
                if latest_funding
                else None
            ),
            "layoff": (
                {
                    "pct": _to_float(latest_layoff.get("Percentage")),
                    "occurred_at": latest_layoff.get("Date"),
                    "scope": latest_layoff.get("Industry"),
                    "source_url": latest_layoff.get("Source"),
                }
                if latest_layoff
                else None
            ),
            "leadership_change": latest_leadership,
            "open_roles_today": open_roles_today,
            "open_roles_60_days_ago": open_roles_60d_ago,
            "tech_stack_detected": crunch.get("tech_stack_detected", []),
            "anti_offshore_public_stance": bool(crunch.get("anti_offshore_public_stance")),
        },
    }


def _to_float(v: Any) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# Task pool I/O
# --------------------------------------------------------------------------


@dataclass
class PoolWriter:
    name: str
    out_dir: Path = OUT_DIR
    tasks: list[dict] = field(default_factory=list)
    seen_hashes: set[str] = field(default_factory=set)
    dropped_dups: int = 0

    def add(self, task: dict) -> bool:
        h = task_hash(task)
        if h in self.seen_hashes:
            self.dropped_dups += 1
            return False
        self.seen_hashes.add(h)
        task.setdefault("source_provenance", {})["dedup_hash"] = h
        self.tasks.append(task)
        return True

    def flush(self) -> Path:
        path = self.out_dir / f"{self.name}.jsonl"
        with path.open("w") as f:
            for t in self.tasks:
                f.write(json.dumps(t, sort_keys=True) + "\n")
        return path


def load_pool(name: str) -> list[dict]:
    path = OUT_DIR / f"{name}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------
# Probe → rubric mapping (re-used by all four authoring modes)
# --------------------------------------------------------------------------


def rubric_for_dimension(
    primary: str,
    *,
    task_type: str,
    has_competitor_brief: bool = False,
    bench_stack_requested: str | None = None,
    bench_count_requested: int | None = None,
) -> dict:
    """Return a rubric block matching the schema's predicate registry.
    Keeps every authoring mode consistent on weights and pass thresholds."""
    if primary == "segment_reasoning":
        return {
            "deterministic_checks": [
                {
                    "name": "segment_label_matches_ground_truth",
                    "predicate": "segment_label_matches_ground_truth",
                    "args": {"field": "segment"},
                    "weight": 0.7,
                },
                {
                    "name": "confidence_above_min",
                    "predicate": "field_gte",
                    "args": {"field": "segment_confidence", "min": 0.6},
                    "weight": 0.3,
                },
            ],
            "judge_checks": [],
            "passing_score": 0.7,
        }

    if primary == "tone_preservation":
        det: list[dict] = [
            {"name": "max_word_count_120", "predicate": "max_word_count",
             "args": {"field": "body", "max": 120}, "weight": 0.10},
            {"name": "no_offshore_cliches", "predicate": "regex_absent",
             "args": {"field": "body", "patterns": ["top talent", "world[- ]class", "A[- ]players?", "rockstars?", "ninjas?"]}, "weight": 0.15},
            {"name": "no_following_up_phrases", "predicate": "regex_absent",
             "args": {"field": "body", "patterns": ["just following up", "circling back", "just checking in"]}, "weight": 0.05},
            {"name": "calendar_link_present", "predicate": "regex_present",
             "args": {"field": "body", "patterns": ["cal\\.com/", "calendly\\.com/"]}, "weight": 0.10},
            {"name": "subject_line_under_60", "predicate": "max_char_count",
             "args": {"field": "subject", "max": 60}, "weight": 0.05},
            {"name": "subject_starts_with_intent_word", "predicate": "regex_present",
             "args": {"field": "subject", "patterns": ["^(Request|Follow-up|Context|Question|Note)"]}, "weight": 0.05},
        ]
        if has_competitor_brief:
            det.append({
                "name": "peer_evidence_source_urls",
                "predicate": "source_url_for_each_quant_claim",
                "args": {"field": "body", "claim_keywords": ["peer", "peers", "competitor", "competitors"]},
                "weight": 0.10,
            })
        judge = [
            {"name": "marker_direct_min_4", "marker": "direct", "min_score": 4, "weight": 0.08},
            {"name": "marker_grounded_min_4", "marker": "grounded", "min_score": 4, "weight": 0.08},
            {"name": "marker_honest_min_4", "marker": "honest", "min_score": 4, "weight": 0.08},
            {"name": "marker_professional_min_4", "marker": "professional", "min_score": 4, "weight": 0.08},
            {"name": "marker_non_condescending_min_4", "marker": "non_condescending", "min_score": 4, "weight": 0.08},
        ]
        # Re-balance det weights to total 0.6 (so judge gets 0.4)
        det_total = sum(c["weight"] for c in det)
        for c in det:
            c["weight"] = round(c["weight"] * (0.6 / det_total), 4)
        return {"deterministic_checks": det, "judge_checks": judge, "passing_score": 0.75}

    if primary == "bench_honesty":
        assert bench_stack_requested and bench_count_requested
        return {
            "deterministic_checks": [
                {"name": "honest_capacity_or_handoff", "predicate": "bench_capacity_not_overcommitted",
                 "args": {"stack": bench_stack_requested, "requested_count": bench_count_requested, "field": "body"},
                 "weight": 0.40},
                {"name": "handoff_flag_or_phased_ramp_language", "predicate": "regex_present_any",
                 "args": {"field": "body", "patterns": ["currently don.?t have", "not on bench", "phased ramp", "route to a colleague", "hand you to"]},
                 "weight": 0.25},
                {"name": "no_overcommit_promise", "predicate": "regex_absent",
                 "args": {"field": "body", "patterns": [
                     f"{bench_count_requested}\\s+{bench_stack_requested}\\s+engineers?\\s+next week",
                     "staff that immediately",
                     f"ready to deploy\\s+{bench_count_requested}\\s+{bench_stack_requested}",
                 ]},
                 "weight": 0.15},
            ],
            "judge_checks": [
                {"name": "marker_honest_min_4", "marker": "honest", "min_score": 4, "weight": 0.10},
                {"name": "marker_professional_min_4", "marker": "professional", "min_score": 4, "weight": 0.10},
            ],
            "passing_score": 0.80,
        }

    if primary == "signal_grounding":
        return {
            "deterministic_checks": [
                {"name": "no_unsourced_funding_amount", "predicate": "regex_absent",
                 "args": {"field": "body", "patterns": ["\\$\\d+(?:\\.\\d+)?\\s*M\\b(?![^\\.]*https?://)"]},
                 "weight": 0.30},
                {"name": "no_aggressive_hiring_on_low_velocity", "predicate": "regex_absent",
                 "args": {"field": "body", "patterns": ["aggressive hiring", "scaling aggressively"]},
                 "weight": 0.25},
                {"name": "no_offshore_cliches", "predicate": "regex_absent",
                 "args": {"field": "body", "patterns": ["top talent", "world[- ]class", "A[- ]players?"]},
                 "weight": 0.15},
            ],
            "judge_checks": [
                {"name": "marker_grounded_min_4", "marker": "grounded", "min_score": 4, "weight": 0.15},
                {"name": "marker_honest_min_4", "marker": "honest", "min_score": 4, "weight": 0.15},
            ],
            "passing_score": 0.75,
        }

    if primary == "gap_framing":
        return {
            "deterministic_checks": [
                {"name": "no_condescending_phrases", "predicate": "regex_absent",
                 "args": {"field": "body", "patterns": [
                     "you'?re missing",
                     "you'?re behind",
                     "you can'?t handle",
                     "your team clearly",
                     "behind the curve",
                 ]},
                 "weight": 0.30},
                {"name": "research_framing_present", "predicate": "regex_present_any",
                 "args": {"field": "body", "patterns": ["peer", "peers", "research", "curious whether", "wanted to ask"]},
                 "weight": 0.20},
            ],
            "judge_checks": [
                {"name": "marker_non_condescending_min_4", "marker": "non_condescending", "min_score": 4, "weight": 0.20},
                {"name": "marker_grounded_min_4", "marker": "grounded", "min_score": 4, "weight": 0.15},
                {"name": "research_framed_min_4", "marker": "research_framed", "min_score": 4, "weight": 0.15},
            ],
            "passing_score": 0.75,
        }

    raise ValueError(f"unknown primary dimension: {primary}")


# --------------------------------------------------------------------------
# Provenance helpers
# --------------------------------------------------------------------------


def make_task(
    *,
    tid: str,
    primary: str,
    task_type: str,
    difficulty: str,
    source_mode: str,
    inputs: dict,
    rubric: dict,
    ground_truth: dict | None = None,
    secondary: list[str] | None = None,
    provenance: dict | None = None,
) -> dict:
    return {
        "task_id": tid,
        "primary_dimension": primary,
        "secondary_dimensions": secondary or [],
        "difficulty": difficulty,
        "source_mode": source_mode,
        "task_type": task_type,
        "input": inputs,
        "ground_truth": ground_truth,
        "rubric": rubric,
        "source_provenance": provenance or {},
    }
