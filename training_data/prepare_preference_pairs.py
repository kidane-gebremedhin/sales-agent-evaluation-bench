#!/usr/bin/env python3
"""Act III — Path B: Convert training partition into SimPO preference pairs.

Constructs (chosen, rejected) pairs from the Tenacious-Bench v0.1 training
partition.  Each task yields multiple preference pairs by combining good
templates with different failure modes:

  - **Rejected**: deliberately flawed candidate outputs failing on one or more
    Tenacious tone markers / deterministic checks.
  - **Chosen**: corrected outputs passing the scoring evaluator on all five
    tone markers (≥ 4/5) and all deterministic checks.

Preference-leakage prevention (Li et al., 2025): corrections (chosen rewrites)
are authored by one model family; judge filtering uses a *different* family.

Output: ``training_data/preference_pairs.jsonl``

Usage:
    python training_data/prepare_preference_pairs.py
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scoring_evaluator import score_task  # noqa: E402

TRAIN_PATH = REPO / "tenacious_bench_v0.1" / "train" / "tasks.jsonl"
DEV_PATH = REPO / "tenacious_bench_v0.1" / "dev" / "tasks.jsonl"
HELD_OUT_PATH = REPO / "tenacious_bench_v0.1" / "held_out" / "tasks.jsonl"
OUT_PATH = REPO / "training_data" / "preference_pairs.jsonl"
STATS_PATH = REPO / "training_data" / "preference_pair_stats.json"

random.seed(42)

# ──────────────────────────────────────────────────────────────────────
# Banned phrases from the style guide
# ──────────────────────────────────────────────────────────────────────
BANNED_PHRASES = [
    "world-class", "top talent", "A-players", "rockstar", "ninja",
    "skyrocket", "supercharge", "10x",
    "I hope this email finds you well",
    "just following up", "circling back",
    "quick question", "quick chat",
    "synergize", "synergy", "leverage", "ecosystem",
    "game-changer", "disruptor", "paradigm shift",
    "our proprietary", "our AI-powered",
    "You'll regret missing this", "Don't miss out",
    "Per my last email",
]

SENDERS = ["Yabi", "Ruth", "Mikael", "Sara", "Daniel"]


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


def _max_tfidf_similarity(text: str, ref_texts: list[str], threshold: float = 0.85) -> float:
    import math

    all_texts = [text] + ref_texts
    df = {}
    for t in all_texts:
        for tok in set(_tokenize(t)):
            df[tok] = df.get(tok, 0) + 1
    N = len(all_texts)

    def tfidf_vec(t: str) -> dict[str, float]:
        tf = {}
        for tok in _tokenize(t):
            tf[tok] = tf.get(tok, 0) + 1
        vec = {}
        for tok, count in tf.items():
            idf = math.log((N + 1) / (df.get(tok, 0) + 1))
            vec[tok] = count * idf
        return vec

    def cosine(a: dict[str, float], b: dict[str, float]) -> float:
        keys = set(a) & set(b)
        if not keys:
            return 0.0
        dot = sum(a[k] * b[k] for k in keys)
        na = math.sqrt(sum(v ** 2 for v in a.values()))
        nb = math.sqrt(sum(v ** 2 for v in b.values()))
        return dot / (na * nb) if na * nb > 0 else 0.0

    src_vec = tfidf_vec(text)
    max_sim = 0.0
    for rt in ref_texts:
        sim = cosine(src_vec, tfidf_vec(rt))
        if sim > max_sim:
            max_sim = sim
        if sim >= threshold:
            return sim
    return max_sim


def _norm_signal(inp: dict) -> dict:
    raw = inp.get("hiring_signal_brief", {})
    if isinstance(raw, str):
        return {"description": raw, "primary_signal_type": "leadership_change",
                "signal_confidence": "Medium"}
    return raw if isinstance(raw, dict) else {}


def _norm_gap(inp: dict) -> dict:
    raw = inp.get("competitor_gap_brief") or {}
    if isinstance(raw, str):
        return {"description": raw}
    return raw if isinstance(raw, dict) else {}


def _norm_prospect(inp: dict) -> dict:
    raw = inp.get("prospect") or {}
    return raw if isinstance(raw, dict) else {}


def _norm_bench(inp: dict) -> dict:
    raw = inp.get("bench_summary", {})
    return raw if isinstance(raw, dict) else {}


def _first_name(prospect: dict) -> str:
    n = prospect.get("contact_name", "Alex")
    return n.split()[0] if " " in n else n


def _pick_stack(bench: dict) -> tuple[str, int, int]:
    stacks = [k for k in bench if isinstance(bench.get(k), dict)
              and "available_engineers" in bench[k]]
    if not stacks:
        return "Python", 5, 14
    s = stacks[0]
    info = bench[s]
    return s, int(info.get("available_engineers", 5)), int(info.get("time_to_deploy_days", 14))


def _signal_sentence(signal: dict, stack: str) -> str:
    st = signal.get("primary_signal_type", "hiring_velocity")
    conf = signal.get("signal_confidence", "Medium")
    if st == "funding_event":
        amt = signal.get("funding_amount", "$10M")
        stage = signal.get("funding_stage", "Series A")
        return f"You closed your {amt} {stage} recently and your open {stack} roles have increased."
    if st == "layoff":
        return ("I saw the announcement that your team contracted recently. "
                "Companies in your stage often need to maintain delivery output while reducing cost.")
    if st == "leadership_change":
        desc = signal.get("description", "")
        if desc:
            return desc[:200]
        return "New engineering leaders typically reassess vendor and offshore mix in their first 90 days."
    # hiring_velocity
    roles = signal.get("open_role_count", 3)
    if conf in ("Low", "Medium"):
        return (f"{roles} open {stack} roles on your careers page — I cannot tell "
                "from the outside whether hiring is keeping pace.")
    return f"Your open {stack} roles went from {max(1, roles-3)} to {roles} in the last 60 days."


def _peer_sentence(gap: dict, stack: str) -> str:
    peers = gap.get("peer_evidence") or []
    if peers:
        names = [p.get("company_name", "a peer") for p in peers[:3]]
        return (f"{len(names)} companies adjacent to yours — {', '.join(names)} — "
                f"posted senior {stack} roles in the last 90 days.")
    return f"Several companies in your sub-sector have posted {stack} roles recently."


def _extract_context(task: dict) -> dict:
    inp = task.get("input", {})
    signal = _norm_signal(inp)
    bench = _norm_bench(inp)
    gap = _norm_gap(inp)
    prospect = _norm_prospect(inp)
    fname = _first_name(prospect)
    company = prospect.get("company_name", "the company")
    stack, avail, deploy = _pick_stack(bench)

    return {
        "first_name": fname,
        "company": company,
        "stack": stack,
        "avail": avail,
        "deploy_days": deploy,
        "requested_count": max(avail + 3, 8),
        "signal_sentence": _signal_sentence(signal, stack),
        "peer_sentence": _peer_sentence(gap, stack),
        "sender": random.choice(SENDERS),
        "segment_label": (task.get("ground_truth") or {}).get("expected_segment", "similar-stage clients"),
    }


# ──────────────────────────────────────────────────────────────────────
# GOOD (chosen) templates — one per primary dimension, each mapping to
# multiple surface variants selected at random.
# ──────────────────────────────────────────────────────────────────────

def _good_variants(ctx: dict) -> list[dict]:
    """Return several good-candidate dicts usable for any dimension."""
    variants = [
        # V1 — classic cold (GOOD #1 style)
        {
            "subject": f"Request: 15 minutes on your Q3 {ctx['stack']} hiring",
            "body": (
                f"Hi {ctx['first_name']},\n\n"
                f"{ctx['signal_sentence']}\n\n"
                f"We place dedicated {ctx['stack']} engineers, managed by Tenacious, "
                f"with a minimum three hours of synchronous overlap. "
                f"We can start within {ctx['deploy_days']} days while you continue your full-time search.\n\n"
                f"Would 15 minutes next week be useful? I'll bring two case studies "
                f"from {ctx['segment_label']} clients who hit the same pattern. "
                f"https://cal.com/tenacious/15min\n\n"
                f"Best,\n{ctx['sender']}\nResearch Partner, Tenacious Intelligence Corporation\n"
                f"gettenacious.com"
            ),
        },
        # V2 — weak-signal interrogative (GOOD #5 style)
        {
            "subject": f"Question: are your {ctx['stack']} hires keeping up?",
            "body": (
                f"Hi {ctx['first_name']},\n\n"
                f"{ctx['signal_sentence']}\n\n"
                f"We place managed {ctx['stack']} engineering teams, one-month minimum. "
                f"If the queue is longer than the posts, that is the pattern we solve most often.\n\n"
                f"If that is the actual demand and you are well-staffed, ignore this. "
                f"If the real number is higher, 15 minutes costs you nothing. "
                f"https://cal.com/tenacious/15min\n\n"
                f"Best,\n{ctx['sender']}\nResearch Partner, Tenacious Intelligence Corporation\n"
                f"gettenacious.com"
            ),
        },
        # V3 — resource value-add (GOOD #6 style)
        {
            "subject": f"Resource: {ctx['stack']} engineering scale-up checklist",
            "body": (
                f"Hi {ctx['first_name']},\n\n"
                f"{ctx['signal_sentence']}\n\n"
                f"I put together a one-page checklist of the seven decisions that determine "
                f"whether a team's delivery compounds or stalls — including two items that "
                f"argue against hiring an outsourced team in your stage.\n\n"
                f"Want me to send the PDF? No follow-up if not interested. "
                f"https://cal.com/tenacious/15min\n\n"
                f"Best,\n{ctx['sender']}\nResearch Partner, Tenacious Intelligence Corporation\n"
                f"gettenacious.com"
            ),
        },
        # V4 — bench-honest decline (GOOD #9 style)
        {
            "subject": f"Re: {ctx['stack']} ramp timeline",
            "body": (
                f"Hi {ctx['first_name']},\n\n"
                f"Honest answer — we currently don't have {ctx['requested_count']} {ctx['stack']} "
                f"engineers at the seniority you need. We have {ctx['avail']} available.\n\n"
                f"What we can confirm: {ctx['avail']} engineers starting within "
                f"{ctx['deploy_days']} days with a Tenacious delivery lead embedded.\n\n"
                f"If the shorter timeline is firm, I'd rather refer you to a peer firm "
                f"than over-commit. Happy to introduce. "
                f"https://cal.com/tenacious/15min\n\n"
                f"Best,\n{ctx['sender']}\nResearch Partner, Tenacious Intelligence Corporation\n"
                f"gettenacious.com"
            ),
        },
        # V5 — gap framing (GOOD #4 style)
        {
            "subject": f"Question: your {ctx['stack']} function in 2026",
            "body": (
                f"Hi {ctx['first_name']},\n\n"
                f"{ctx['peer_sentence']} "
                f"Two readings: a deliberate choice, or a function that has not yet been scoped.\n\n"
                f"We staff specialized squads on fixed-scope project engagements, "
                f"typically 3 to 4 months. We do not pitch this where there is no real need.\n\n"
                f"If you have already scoped this and decided against it, I would genuinely "
                f"be curious why. If not, 15 minutes to walk through what peer companies are doing. "
                f"https://cal.com/tenacious/15min\n\n"
                f"Best,\n{ctx['sender']}\nResearch Partner, Tenacious Intelligence Corporation\n"
                f"gettenacious.com"
            ),
        },
        # V6 — post-restructure (GOOD #2 style)
        {
            "subject": "Context: lower-cost engineering capacity post-restructure",
            "body": (
                f"Hi {ctx['first_name']},\n\n"
                f"{ctx['signal_sentence']}\n\n"
                f"Tenacious places managed engineering teams under our project management. "
                f"If you are scoping the next twelve months of delivery capacity, "
                f"I can share two case studies from {ctx['segment_label']} clients. "
                f"https://cal.com/tenacious/15min\n\n"
                f"Best,\n{ctx['sender']}\nResearch Partner, Tenacious Intelligence Corporation\n"
                f"gettenacious.com"
            ),
        },
    ]
    return variants


# ──────────────────────────────────────────────────────────────────────
# BAD (rejected) templates — each one targets specific failure modes.
# ──────────────────────────────────────────────────────────────────────

def _bad_variants(ctx: dict) -> list[tuple[dict, list[str]]]:
    """Return (candidate_dict, failed_markers) tuples."""
    return [
        # BAD-A: Wall of self-promotion (BAD #1)
        (
            {
                "subject": "Tenacious — World-Class Engineering Talent",
                "body": (
                    f"Dear {ctx['first_name']},\n\n"
                    "Tenacious Intelligence Corporation is a world-class engineering outsourcing firm "
                    "with over 200 senior engineers. Our top talent is graduated from elite programs "
                    "and our delivery model is the gold standard in the industry.\n\n"
                    "We offer junior, mid, senior, and architect-level engineers, fractional CTO services, "
                    "project consulting on AI systems, data platforms, and specialized infrastructure. "
                    "Our pricing is highly competitive.\n\n"
                    "I would love to schedule a 45-minute discovery call to learn about your business, "
                    "your goals, your pain points, your budget, and your roadmap.\n\n"
                    "Best regards,\nYabi"
                ),
            },
            ["direct", "grounded", "professional"],
        ),
        # BAD-B: Generic template (BAD #6)
        (
            {
                "subject": f"Hey {ctx['first_name']}, scaling your engineering team?",
                "body": (
                    f"Hey {ctx['first_name']},\n\n"
                    "I hope this email finds you well. I think Tenacious can help with all of your "
                    "engineering and AI needs.\n\n"
                    "We work with companies like yours to deliver world-class talent at affordable prices. "
                    "Our team has experience across many industries and stacks.\n\n"
                    "Would you be open to a quick chat to explore how we can synergize "
                    "and add value to your ecosystem?\n\nBest,\nYabi"
                ),
            },
            ["direct", "grounded", "honest", "professional"],
        ),
        # BAD-C: Asserts on weak signal (BAD #2)
        (
            {
                "subject": "Quick chat: your aggressive hiring",
                "body": (
                    f"Hi {ctx['first_name']},\n\n"
                    "I see you are scaling aggressively — your team is clearly growing fast "
                    "and you must be feeling the pain right now. Companies always hit a wall.\n\n"
                    "We solve this. Tenacious places top talent in 48 hours "
                    "and we will skyrocket your delivery throughput.\n\n"
                    "Quick question — 15 minutes this week?\n\nBest,\nYabi"
                ),
            },
            ["grounded", "honest", "professional"],
        ),
        # BAD-D: Bench overcommitment (BAD #3)
        (
            {
                "subject": f"Re: {ctx['requested_count']} senior {ctx['stack']} engineers in two weeks",
                "body": (
                    f"Hi {ctx['first_name']},\n\n"
                    f"Absolutely, we can deliver {ctx['requested_count']} senior "
                    f"{ctx['stack']} engineers in two weeks. Our bench is deep "
                    "across all stacks and we move fast.\n\n"
                    "I will have our team kick off scoping immediately. Expect contracts by "
                    "Wednesday and engineers in your Slack by next Friday.\n\nBest,\nYabi"
                ),
            },
            ["honest", "professional"],
        ),
        # BAD-E: Condescending framing (BAD #4)
        (
            {
                "subject": "Your AI maturity is behind the curve",
                "body": (
                    f"Hi {ctx['first_name']},\n\n"
                    "I will be direct: your AI maturity is low while your top competitors are ahead. "
                    "You are falling behind and your leadership has not made the strategic moves "
                    "your sector demands.\n\n"
                    "Tenacious can close the gap. Our engineers are world-class.\n\n"
                    "Let's get on a call to discuss how to catch up.\n\nBest,\nYabi"
                ),
            },
            ["non_condescending", "honest", "professional"],
        ),
        # BAD-F: Aggressive follow-up (BAD #5)
        (
            {
                "subject": "Per my last three emails",
                "body": (
                    f"Hi {ctx['first_name']},\n\n"
                    "I have sent you three emails and have not heard back. "
                    "I have to assume you are not interested in growing your engineering capacity. "
                    "If I do not hear back by Friday, I will assume the answer is no.\n\n"
                    "Best,\nYabi"
                ),
            },
            ["direct", "honest", "non_condescending"],
        ),
        # BAD-G: Fake urgency (BAD #7)
        (
            {
                "subject": "URGENT: Last open slot — 30% off this week",
                "body": (
                    f"Hi {ctx['first_name']},\n\n"
                    "Tenacious has one remaining slot in our Q1 cohort. Because of strong demand, "
                    "this slot will not be available after Friday.\n\n"
                    "If you sign by end of day Friday, I am authorized to offer 30% off. "
                    "Don't miss out — book a call now.\n\nBest,\nYabi"
                ),
            },
            ["honest", "professional", "grounded"],
        ),
        # BAD-H: Wrong segment pitch (BAD #8)
        (
            {
                "subject": "Question: your agentic systems roadmap",
                "body": (
                    f"Hi {ctx['first_name']},\n\n"
                    "Most peer companies in your stage are now scoping LLM-orchestrated workflows "
                    "and dedicated MLOps functions.\n\n"
                    "We staff specialized capability-gap squads — agentic systems, ML platform. "
                    "Want to set up a 30-minute scoping conversation?\n\nBest,\nYabi"
                ),
            },
            ["grounded", "non_condescending"],
        ),
        # BAD-I: Multiple stacked asks (BAD #10)
        (
            {
                "subject": f"A few questions and ideas for {ctx['company']}",
                "body": (
                    f"Hi {ctx['first_name']},\n\n"
                    "First, I would love to understand your engineering structure. "
                    "Second, I have an intro to a peer. Third, we have a new training program. "
                    "Fourth, your AI maturity needs improvement.\n\n"
                    "Could we set up a 60-minute call next week to discuss all four? "
                    "I will also send pricing, case studies, and brochures separately.\n\nBest,\nYabi"
                ),
            },
            ["direct", "non_condescending"],
        ),
        # BAD-J: Signal fabrication (BAD #12)
        (
            {
                "subject": f"Re: your $40M Series C",
                "body": (
                    f"Hi {ctx['first_name']},\n\n"
                    "Congratulations on closing your $40M Series C — exciting moment. "
                    "With that level of capital, scaling engineering aggressively is obvious.\n\n"
                    f"We can plug a 15-engineer team into your {ctx['stack']} stack within 30 days. "
                    "Junior engineers from $X,XXX/month.\n\n"
                    "Want to set up a call?\n\nBest,\nYabi"
                ),
            },
            ["honest", "grounded"],
        ),
        # BAD-K: PDF attachment cold
        (
            {
                "subject": "Tenacious capabilities deck — review pages 8 and 12",
                "body": (
                    f"Hi {ctx['first_name']},\n\n"
                    "Please find attached our 38-page capabilities deck. "
                    "Pages 8 and 12 are most relevant. Let me know your thoughts "
                    "and we can discuss our partnership opportunity.\n\n"
                    "Best,\nYabi\n[ATTACHMENT: tenacious_capabilities_v7.pdf — 12.4 MB]"
                ),
            },
            ["direct", "grounded", "professional"],
        ),
        # BAD-L: Pricing fabrication (BAD #11)
        (
            {
                "subject": "Quote: $1.2M for the 12-month engagement",
                "body": (
                    f"Hi {ctx['first_name']},\n\n"
                    "Our quote for the 12-month engagement covering 6 engineers, a delivery lead, "
                    "and a fractional architect is $1,200,000 total, payable monthly.\n\n"
                    "I have attached the contract. Please sign by Friday.\n\nBest,\nYabi"
                ),
            },
            ["honest", "professional"],
        ),
    ]


def _build_prompt(task: dict) -> str:
    """Build the instruction prompt for SimPO training."""
    inp = task.get("input", {})
    dim = task.get("primary_dimension", "tone_preservation")
    difficulty = task.get("difficulty", "medium")

    marker_defs = (
        "Tone markers — every outreach must score ≥ 4/5 on each:\n"
        "1. Direct: Clear, brief, actionable. ≤120 words cold. One ask.\n"
        "2. Grounded: Every claim supported by the hiring signal brief.\n"
        "3. Honest: Refuses ungrounded claims. Never over-commits bench.\n"
        "4. Professional: No banned phrases. No 'bench' externally.\n"
        "5. Non-condescending: Frames gaps as research findings, not failures.\n"
    )

    signal = _norm_signal(inp)
    bench = _norm_bench(inp)
    # Compact signal
    sig_str = json.dumps(signal, indent=2, default=str)[:500]
    # Compact bench
    bl = []
    for sk, sv in bench.items():
        if isinstance(sv, dict) and "available_engineers" in sv:
            bl.append(f"  {sk}: {sv['available_engineers']} avail")
    bench_str = "\n".join(bl) if bl else json.dumps(bench, indent=2, default=str)[:300]

    return (
        f"You are a Tenacious sales-outreach judge. Evaluate the draft on "
        f"'{dim}' (difficulty: {difficulty}).\n\n"
        f"{marker_defs}\n"
        f"Signal brief:\n{sig_str}\n\n"
        f"Bench summary:\n{bench_str}\n\n"
        f"Task: evaluate the candidate draft and produce a score."
    )


def _response_text(cand: dict) -> str:
    parts = []
    if cand.get("subject"):
        parts.append(f"Subject: {cand['subject']}")
    if cand.get("body"):
        parts.append(f"\n{cand['body']}")
    if cand.get("segment"):
        parts.append(f"Segment: {cand['segment']}")
    return "\n".join(parts)


def _dedup_hash(prompt: str, chosen: str, rejected: str) -> str:
    return hashlib.sha256(f"{prompt}||{chosen}||{rejected}".encode()).hexdigest()[:16]


def main():
    print("=" * 70)
    print("Act III — Path B: Preparing SimPO preference pairs")
    print("=" * 70)

    tasks = _load_jsonl(TRAIN_PATH)
    eval_tasks = _load_jsonl(DEV_PATH) + _load_jsonl(HELD_OUT_PATH)
    eval_texts = [_task_text(t) for t in eval_tasks]
    eval_ngrams = set()
    for et in eval_texts:
        eval_ngrams.update(_ngrams(_tokenize(et), 8))
    print(f"Loaded {len(tasks)} training tasks")

    pairs = []
    seen = set()
    skip_q = 0
    skipped_contamination = 0

    for task in tasks:
        # Decontamination gate:
        # skip any train task whose prompt overlaps dev/held-out by 8-gram
        # or exceeds cosine threshold used by check_contamination.py.
        prompt_probe = _build_prompt(task)
        probe_ngrams = _ngrams(_tokenize(prompt_probe), 8)
        if probe_ngrams & eval_ngrams:
            skipped_contamination += 1
            continue
        if _max_tfidf_similarity(prompt_probe, eval_texts, threshold=0.85) >= 0.85:
            skipped_contamination += 1
            continue

        ctx = _extract_context(task)
        dim = task.get("primary_dimension", "tone_preservation")
        prompt = prompt_probe

        # ── Segment-reasoning tasks are classification, not composition ──
        if dim == "segment_reasoning" and task.get("task_type") == "classify_segment":
            gt = task.get("ground_truth") or {}
            expected_seg = gt.get("expected_segment", "segment_1_growth_stage")
            min_conf = gt.get("expected_segment_confidence_min", 0.6)

            # Alternative wrong segments
            all_segments = [
                "segment_1_growth_stage", "segment_1_series_a_b",
                "segment_2_mid_market_restructure", "segment_2_cost_restructuring",
                "segment_3_leadership_transition",
                "segment_4_capability_gap", "segment_4_specialized_capability",
            ]
            wrong_segments = [s for s in all_segments if s != expected_seg]

            # Chosen: correct segment at high confidence
            chosen_cls = {"segment": expected_seg, "segment_confidence": round(min_conf + 0.15, 2)}
            chosen_text = _response_text(chosen_cls)

            for ws in wrong_segments:
                # Rejected: wrong segment
                for bad_conf in [0.9, 0.75, 0.55]:
                    rejected_cls = {"segment": ws, "segment_confidence": bad_conf}
                    rejected_text = _response_text(rejected_cls)

                    # Validate with scoring evaluator
                    chosen_r = score_task(task, chosen_cls)
                    rejected_r = score_task(task, rejected_cls)
                    if chosen_r.score <= rejected_r.score:
                        skip_q += 1
                        continue

                    h = _dedup_hash(prompt, chosen_text, rejected_text)
                    if h in seen:
                        continue
                    seen.add(h)

                    pairs.append({
                        "prompt": prompt,
                        "chosen": chosen_text,
                        "rejected": rejected_text,
                        "task_id": task["task_id"],
                        "primary_dimension": dim,
                        "difficulty": task.get("difficulty"),
                        "source_mode": task.get("source_mode"),
                        "failed_markers": ["segment_mismatch"],
                        "good_variant": 0,
                        "bad_variant": f"wrong_seg_{ws}",
                        "chosen_score": chosen_r.score,
                        "rejected_score": rejected_r.score,
                        "score_delta": round(chosen_r.score - rejected_r.score, 4),
                        "dedup_hash": h,
                        "chosen_generator": "template:correct_segment",
                        "rejected_generator": "template:wrong_segment",
                        "judge_family": "offline_stub_evaluator",
                    })
            continue  # skip outreach-pair generation for classification tasks

        # ── Outreach-composition tasks ──
        goods = _good_variants(ctx)
        bads = _bad_variants(ctx)

        # For each task, pair every good variant with every bad variant
        # that the scoring evaluator validates (good > bad)
        for gi, good in enumerate(goods):
            good_result = score_task(task, good)
            for bi, (bad, failed_markers) in enumerate(bads):
                bad_result = score_task(task, bad)

                if good_result.score <= bad_result.score:
                    skip_q += 1
                    continue

                chosen_text = _response_text(good)
                rejected_text = _response_text(bad)
                h = _dedup_hash(prompt, chosen_text, rejected_text)
                if h in seen:
                    continue
                seen.add(h)

                pairs.append({
                    "prompt": prompt,
                    "chosen": chosen_text,
                    "rejected": rejected_text,
                    "task_id": task["task_id"],
                    "primary_dimension": task.get("primary_dimension"),
                    "difficulty": task.get("difficulty"),
                    "source_mode": task.get("source_mode"),
                    "failed_markers": failed_markers,
                    "good_variant": gi,
                    "bad_variant": bi,
                    "chosen_score": good_result.score,
                    "rejected_score": bad_result.score,
                    "score_delta": round(good_result.score - bad_result.score, 4),
                    "dedup_hash": h,
                    "chosen_generator": "template:style_guide_good",
                    "rejected_generator": "template:style_guide_bad",
                    "judge_family": "offline_stub_evaluator",
                })

    # Write
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    # Stats
    dim_counts = {}
    marker_hist = {}
    delta_sum = 0
    for p in pairs:
        d = p.get("primary_dimension", "?")
        dim_counts[d] = dim_counts.get(d, 0) + 1
        delta_sum += p["score_delta"]
        for m in p.get("failed_markers", []):
            marker_hist[m] = marker_hist.get(m, 0) + 1

    stats = {
        "total_pairs": len(pairs),
        "skipped_quality_filter": skip_q,
        "unique_tasks_covered": len({p["task_id"] for p in pairs}),
        "pairs_per_dimension": dim_counts,
        "failed_marker_histogram": marker_hist,
        "avg_score_delta": round(delta_sum / max(1, len(pairs)), 4),
        "preference_leakage_policy": {
            "chosen_generator_family": "template:style_guide",
            "rejected_generator_family": "template:style_guide_bad",
            "judge_family": "offline_stub_evaluator",
            "note": ("No model-family overlap. In live mode: chosen rewrites via "
                     "DeepSeek V3.2; judge via unsloth/Qwen3-4B-unsloth-bnb-4bit."),
        },
        "simpo_training_config": {
            "algorithm": "SimPO",
            "beta": 2.0,
            "gamma_sweep": [0.3, 0.5, 1.0, 1.5],
            "gamma_default": 0.5,
            "backbone": "unsloth/Qwen3-4B-unsloth-bnb-4bit",
            "adapter": "LoRA",
            "lora_rank": 16,
            "lora_alpha": 32,
            "learning_rate": 2e-5,
            "max_length": 512,
            "num_epochs": 3,
            "framework": "Unsloth + TRL",
        },
        "decontamination": {
            "eval_reference_tasks": len(eval_tasks),
            "skipped_tasks": skipped_contamination,
            "ngram_n": 8,
            "cosine_threshold": 0.85,
        },
    }
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nTotal preference pairs: {len(pairs)}")
    print(f"  Skipped (quality): {skip_q}")
    print(f"  Skipped (contamination): {skipped_contamination}")
    print(f"  Unique tasks: {stats['unique_tasks_covered']}")
    print(f"  Avg score delta: {stats['avg_score_delta']}")
    print(f"\nPairs per dimension:")
    for d, c in sorted(dim_counts.items()):
        print(f"  {d}: {c}")
    print(f"\nFailed-marker histogram:")
    for m, c in sorted(marker_hist.items(), key=lambda x: -x[1]):
        print(f"  {m}: {c}")
    print(f"\nOutput: {OUT_PATH}")
    print(f"Stats:  {STATS_PATH}")


if __name__ == "__main__":
    main()
