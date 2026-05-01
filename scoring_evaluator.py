"""Tenacious-Bench v0.1 — machine-verifiable scoring evaluator.

Reads a Tenacious-Bench task plus a candidate output and returns a
numerical score in [0, 1] with no human in the loop. Deterministic
predicates run locally; judge predicates default to a deterministic
stub for offline runs and route to an OpenRouter dev-tier model when
``TB_USE_LIVE_JUDGE=1`` is set in the environment.

Usage as a module
-----------------
    from scoring_evaluator import score_task
    result = score_task(task, candidate_output)
    # result.passed: bool
    # result.score: float in [0, 1]
    # result.breakdown: list of per-check rows

CLI
---
    python scoring_evaluator.py --schema schema.json --self-test
    # Runs the three example tasks against canned good and bad
    # candidate outputs and prints a summary table.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------


@dataclass
class CheckResult:
    name: str
    kind: str  # "deterministic" | "judge"
    passed: bool
    weight: float
    detail: str = ""
    judge_score: int | None = None


@dataclass
class TaskResult:
    task_id: str
    score: float
    passing_score: float
    passed: bool
    breakdown: list[CheckResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "score": self.score,
            "passing_score": self.passing_score,
            "passed": self.passed,
            "breakdown": [asdict(c) for c in self.breakdown],
        }


# --------------------------------------------------------------------------
# Deterministic predicates
# --------------------------------------------------------------------------


def _get_field(candidate: dict[str, Any], field_name: str) -> str:
    """Tolerant lookup. Defaults to empty string if the field is missing."""
    if field_name in candidate and candidate[field_name] is not None:
        v = candidate[field_name]
        return v if isinstance(v, str) else json.dumps(v)
    return ""


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def pred_max_word_count(candidate: dict, args: dict, **_) -> tuple[bool, str]:
    text = _get_field(candidate, args["field"])
    n = _word_count(text)
    return n <= int(args["max"]), f"word_count={n} max={args['max']}"


def pred_max_char_count(candidate: dict, args: dict, **_) -> tuple[bool, str]:
    text = _get_field(candidate, args["field"])
    return len(text) <= int(args["max"]), f"char_count={len(text)} max={args['max']}"


def pred_regex_absent(candidate: dict, args: dict, **_) -> tuple[bool, str]:
    text = _get_field(candidate, args["field"])
    hits = [p for p in args["patterns"] if re.search(p, text, re.IGNORECASE | re.MULTILINE)]
    return (len(hits) == 0), (f"hits={hits}" if hits else "no banned patterns matched")


def pred_regex_present(candidate: dict, args: dict, **_) -> tuple[bool, str]:
    """Pass if EVERY pattern matches (used for required signature/structural elements)."""
    text = _get_field(candidate, args["field"])
    misses = [p for p in args["patterns"] if not re.search(p, text, re.IGNORECASE | re.MULTILINE)]
    return (len(misses) == 0), (f"missing={misses}" if misses else "all required patterns matched")


def pred_regex_present_any(candidate: dict, args: dict, **_) -> tuple[bool, str]:
    """Pass if ANY pattern matches (used for handoff or honesty signals)."""
    text = _get_field(candidate, args["field"])
    hits = [p for p in args["patterns"] if re.search(p, text, re.IGNORECASE | re.MULTILINE)]
    return (len(hits) > 0), (f"matched={hits}" if hits else f"none of {args['patterns']} matched")


def pred_field_gte(candidate: dict, args: dict, **_) -> tuple[bool, str]:
    v = candidate.get(args["field"])
    try:
        return (v is not None and float(v) >= float(args["min"])), f"{args['field']}={v} min={args['min']}"
    except (TypeError, ValueError):
        return False, f"{args['field']}={v!r} not numeric"


def pred_segment_label_matches_ground_truth(
    candidate: dict, args: dict, *, task: dict, **_
) -> tuple[bool, str]:
    expected = task.get("ground_truth", {}).get("expected_segment")
    actual = candidate.get(args.get("field", "segment"))
    return (expected is not None and actual == expected), f"expected={expected} actual={actual}"


_QUANT_CLAIM_RE = re.compile(r"\$\d|\d+\s?(?:%|peer|peers|companies|engineers|roles|round)", re.IGNORECASE)


def pred_source_url_for_each_quant_claim(
    candidate: dict, args: dict, *, task: dict, **_
) -> tuple[bool, str]:
    """Heuristic: if the body asserts a quantitative claim about peers/competitors/funding,
    the brief must contain at least one peer_evidence entry with a source_url, OR the
    body must include a URL itself."""
    text = _get_field(candidate, args["field"])
    has_quant = bool(_QUANT_CLAIM_RE.search(text))
    if not has_quant:
        return True, "no quantitative claim detected"

    brief = task.get("input", {}).get("competitor_gap_brief") or {}
    peers = brief.get("peer_evidence") or []
    has_brief_source = any(p.get("source_url") for p in peers)
    has_inline_url = bool(re.search(r"https?://", text))
    ok = has_brief_source or has_inline_url
    return ok, f"has_brief_source={has_brief_source} has_inline_url={has_inline_url}"


_NUM_WORD = r"(?:\d+|two|three|four|five|six|seven|eight|nine|ten)"


def pred_bench_capacity_not_overcommitted(
    candidate: dict, args: dict, *, task: dict, **_
) -> tuple[bool, str]:
    """Pass if either (a) bench has enough engineers in the requested stack, or
    (b) the body does not contain an *affirmative* overcommit pattern.
    A negated mention ("we don't have four Rust engineers") does not fail
    the check — the body is still being honest about the gap."""
    stack = args["stack"]
    requested = int(args["requested_count"])
    bench = task.get("input", {}).get("bench_summary", {}) or {}
    available = int((bench.get(stack) or {}).get("available_engineers", 0))

    if available >= requested:
        return True, f"bench_ok available={available} requested={requested}"

    text = _get_field(candidate, args["field"])

    # Affirmative-overcommit patterns. Each must be free of an immediately
    # preceding negation cue ("don't", "not", "no", "without", "rather than").
    affirmative_patterns = [
        rf"\b{_NUM_WORD}\s+{re.escape(stack)}\s+engineers?\s+(?:ready|available|on the bench|next week|immediately)",
        rf"\bstaff (?:that|the squad|four|five) (?:immediately|right away|next week)\b",
        rf"\bwe(?:'ll| will) have\s+{_NUM_WORD}\s+{re.escape(stack)}\b",
        rf"\bready to deploy\s+{_NUM_WORD}\s+{re.escape(stack)}\b",
        rf"\babsolutely[^.]*staff that\b",
    ]
    negation_lookback = re.compile(
        r"\b(?:don'?t|do not|cannot|can'?t|won'?t|will not|no|not|without|rather than|don'?t have)\b",
        re.IGNORECASE,
    )

    for pat in affirmative_patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            window = text[max(0, m.start() - 40) : m.start()]
            if negation_lookback.search(window):
                continue
            return False, f"affirmative overcommit: {m.group(0)!r}"

    return True, f"bench_short available={available}, no affirmative overcommit"


PREDICATES: dict[str, Callable[..., tuple[bool, str]]] = {
    "max_word_count": pred_max_word_count,
    "max_char_count": pred_max_char_count,
    "regex_absent": pred_regex_absent,
    "regex_present": pred_regex_present,
    "regex_present_any": pred_regex_present_any,
    "field_gte": pred_field_gte,
    "segment_label_matches_ground_truth": pred_segment_label_matches_ground_truth,
    "source_url_for_each_quant_claim": pred_source_url_for_each_quant_claim,
    "bench_capacity_not_overcommitted": pred_bench_capacity_not_overcommitted,
}


# --------------------------------------------------------------------------
# Judge — five tone markers
# --------------------------------------------------------------------------

# Heuristic anti-markers used by the offline stub judge.
_DISALLOWED_PHRASES = [
    r"\btop talent\b",
    r"\bworld[- ]class\b",
    r"\bA[- ]players?\b",
    r"\brockstars?\b",
    r"\bninjas?\b",
    r"\bjust following up\b",
    r"\bcircling back\b",
    r"\bjust checking in\b",
    r"\bclearly behind\b",
    r"\bcan't handle this\b",
]

_GROUNDING_HINTS = [r"https?://", r"\bSeries [ABCD]\b", r"\b\d+\s+open\b", r"\bpeer\b"]


def stub_marker_score(marker: str, body: str) -> int:
    """Cheap deterministic stand-in for an LLM judge. Used when
    TB_USE_LIVE_JUDGE is unset, so the self-test runs offline.
    Heuristics are intentionally simple — the live judge replaces them
    in production runs."""
    body_l = body.lower()
    hits_disallowed = sum(1 for p in _DISALLOWED_PHRASES if re.search(p, body, re.IGNORECASE))

    if marker == "direct":
        return 5 if 20 <= _word_count(body) <= 130 else 3
    if marker == "grounded":
        hits = sum(1 for p in _GROUNDING_HINTS if re.search(p, body, re.IGNORECASE))
        return min(5, 2 + hits)
    if marker == "honest":
        return max(1, 5 - hits_disallowed)
    if marker == "professional":
        return max(1, 5 - hits_disallowed)
    if marker == "non_condescending":
        bad = sum(
            1
            for p in (r"\byou.?re missing\b", r"\bbehind the curve\b", r"\bcan't handle\b")
            if re.search(p, body, re.IGNORECASE)
        )
        return max(1, 5 - 2 * bad)
    if marker == "research_framed":
        return 5 if "peer" in body_l or "research" in body_l else 3
    return 3


def live_marker_score(marker: str, body: str, task: dict) -> int:
    """Calls an OpenRouter dev-tier judge. Only invoked when
    TB_USE_LIVE_JUDGE=1. Returns 1–5."""
    import urllib.request

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("TB_USE_LIVE_JUDGE=1 but OPENROUTER_API_KEY is unset")
    model = os.environ.get("TB_JUDGE_MODEL", "unsloth/Qwen3-4B-unsloth-bnb-4bit")
    prompt = (
        f"You are scoring a Tenacious sales outreach draft on the marker '{marker}'. "
        f"Return a single integer 1-5. Marker definitions: see Tenacious style guide. "
        f"Draft body:\n---\n{body}\n---\nScore (1-5):"
    )
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 5,
        "temperature": 0.0,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    text = data["choices"][0]["message"]["content"]
    m = re.search(r"[1-5]", text)
    return int(m.group(0)) if m else 3


def marker_score(marker: str, body: str, task: dict) -> int:
    if os.environ.get("TB_USE_LIVE_JUDGE") == "1":
        return live_marker_score(marker, body, task)
    return stub_marker_score(marker, body)


# --------------------------------------------------------------------------
# Scoring loop
# --------------------------------------------------------------------------


def score_task(task: dict, candidate: dict) -> TaskResult:
    rubric = task["rubric"]
    breakdown: list[CheckResult] = []
    weighted_pass = 0.0
    total_weight = 0.0

    for check in rubric.get("deterministic_checks", []):
        pred_name = check["predicate"]
        pred = PREDICATES.get(pred_name)
        weight = float(check.get("weight", 1.0))
        total_weight += weight
        if pred is None:
            breakdown.append(
                CheckResult(check["name"], "deterministic", False, weight, f"unknown predicate {pred_name!r}")
            )
            continue
        ok, detail = pred(candidate, check.get("args", {}), task=task)
        if ok:
            weighted_pass += weight
        breakdown.append(CheckResult(check["name"], "deterministic", ok, weight, detail))

    body = candidate.get("body", "") or candidate.get("text", "") or ""
    for check in rubric.get("judge_checks", []):
        weight = float(check.get("weight", 1.0))
        total_weight += weight
        score_int = marker_score(check["marker"], body, task)
        ok = score_int >= int(check["min_score"])
        if ok:
            weighted_pass += weight
        breakdown.append(
            CheckResult(check["name"], "judge", ok, weight, f"score={score_int} min={check['min_score']}", judge_score=score_int)
        )

    score = (weighted_pass / total_weight) if total_weight > 0 else 0.0
    passing = float(rubric.get("passing_score", 0.7))
    return TaskResult(
        task_id=task["task_id"],
        score=round(score, 4),
        passing_score=passing,
        passed=(score >= passing),
        breakdown=breakdown,
    )


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------


def _good_candidate_for(task_id: str) -> dict:
    """Hand-built candidate that should pass the rubric."""
    if task_id == "TB-0001":
        return {"segment": "segment_2_mid_market_restructure", "segment_confidence": 0.78}
    if task_id == "TB-0002":
        body = (
            "Maya,\n\n"
            "Three of your peers have posted AI-platform engineer roles in the last 90 days "
            "(see https://example/peer-a and https://example/peer-b). Your team has 8 open "
            "engineering roles today and no named ML lead in the public team page.\n\n"
            "We run dedicated ML squads — peer companies typically scope this as a 2-engineer "
            "embed, with our architect joining for the first month. Worth 15 minutes next "
            "Tuesday or Wednesday to walk you through the shape? https://cal.com/tenacious/15min\n\n"
            "Maya\nResearch Partner, Tenacious Intelligence Corporation\ngettenacious.com"
        )
        return {"subject": "Question on your AI platform roadmap", "body": body}
    if task_id == "TB-0003":
        body = (
            "Marc,\n\n"
            "Honest answer first — we currently don't have four Rust engineers on the bench. "
            "We have three Go engineers and seven Python engineers, none of whom would be a "
            "responsible swap for a payments service. I'd rather route you to a colleague "
            "who can speak to a phased ramp using our infra team while you recruit Rust "
            "in-house, than promise capacity we don't have.\n\n"
            "Want me to make that intro?\n\nMaya"
        )
        return {"body": body}
    raise KeyError(task_id)


def _bad_candidate_for(task_id: str) -> dict:
    """Hand-built candidate that should fail the rubric."""
    if task_id == "TB-0001":
        return {"segment": "segment_1_series_a_b", "segment_confidence": 0.82}
    if task_id == "TB-0002":
        body = (
            "Hey there! Just checking in — wanted to reach out about your team. "
            "We work with world-class top talent and rockstars who can scale your "
            "AI strategy aggressively. You're clearly behind the curve on AI; let "
            "us help you fix that. Happy to chat anytime!"
        )
        return {"subject": "Quick question (just a quick chat)", "body": body}
    if task_id == "TB-0003":
        body = (
            "Marc, absolutely — we can staff that immediately. We'll have four Rust "
            "engineers ready to deploy next week, all senior, all world-class. Talk soon."
        )
        return {"body": body}
    raise KeyError(task_id)


def run_self_test(schema_path: Path) -> int:
    schema = json.loads(schema_path.read_text())
    examples = schema["example_tasks"]

    rows: list[dict] = []
    for task in examples:
        tid = task["task_id"]
        good = _good_candidate_for(tid)
        bad = _bad_candidate_for(tid)
        good_r = score_task(task, good)
        bad_r = score_task(task, bad)
        rows.append({
            "task_id": tid,
            "primary_dimension": task["primary_dimension"],
            "good_score": good_r.score,
            "good_passed": good_r.passed,
            "bad_score": bad_r.score,
            "bad_passed": bad_r.passed,
        })

    print(f"{'task':<10}{'dim':<22}{'good':<14}{'bad':<14}{'discriminates':<14}")
    failures = 0
    for r in rows:
        discriminates = bool(r["good_passed"]) and not bool(r["bad_passed"])
        if not discriminates:
            failures += 1
        print(
            f"{r['task_id']:<10}{r['primary_dimension']:<22}"
            f"{r['good_score']:<6.3f}{'pass' if r['good_passed'] else 'fail':<8}"
            f"{r['bad_score']:<6.3f}{'pass' if r['bad_passed'] else 'fail':<8}"
            f"{'OK' if discriminates else 'FAIL':<14}"
        )

    print()
    if failures:
        print(f"SELF-TEST FAILED: {failures} of {len(rows)} tasks did not discriminate good from bad")
        return 1
    print(f"SELF-TEST OK: all {len(rows)} example tasks discriminated good from bad candidates")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    p.add_argument("--schema", type=Path, default=REPO_ROOT / "schema.json")
    p.add_argument("--self-test", action="store_true",
                   help="Run the three example tasks against canned good/bad candidates.")
    p.add_argument("--task-id", help="Score a single task; reads candidate JSON on stdin.")
    args = p.parse_args(argv)

    if args.self_test:
        return run_self_test(args.schema)

    if args.task_id:
        schema = json.loads(args.schema.read_text())
        task = next((t for t in schema["example_tasks"] if t["task_id"] == args.task_id), None)
        if task is None:
            print(f"task {args.task_id} not found in {args.schema}", file=sys.stderr)
            return 2
        candidate = json.load(sys.stdin)
        result = score_task(task, candidate)
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.passed else 1

    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
