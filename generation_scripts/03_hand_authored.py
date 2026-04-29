"""Hand-authored adversarial tasks (~15% of Tenacious-Bench v0.1).

Per the challenge spec, these are written by the trainee (me) targeting
edge cases the synthesis pipeline misses. They carry the most originality
weight at grading. Each task picks a Tenacious-specific failure mode
that a generic τ²-Bench-tuned agent would not anticipate, and gives the
agent a brief that *looks* clean but has a buried trap.

The tasks below were authored on 2026-04-29 against the audit_memo's
five gap dimensions. They target failure modes seen or anticipated in
Week 10 trace pool but not exhaustively covered by the programmatic
sweeps.
"""

from __future__ import annotations

import json
from pathlib import Path

from common import (
    PoolWriter,
    make_task,
    rubric_for_dimension,
    task_id,
)


# Each adversarial task is a dict spec; we expand to full schema below.
ADVERSARIAL_SPECS = [
    # ------------------------------------------------------------------
    # Segment-reasoning adversarials
    # ------------------------------------------------------------------
    {
        "tag": "layoff_pct_sub_threshold",
        "primary": "segment_reasoning", "task_type": "classify_segment", "difficulty": "adversarial",
        "probes": ["P-0001", "P-0004"],
        "trace_id": "trp1_week10_conversion_engine_b3f5aa034a4f",
        "input": {
            "prospect_meta": {"domain": "synth-adv-001.example", "persona": "VP Engineering", "timezone": "America/New_York"},
            "hiring_signal_brief": {
                "funding": {"round": "series_b", "amount_usd": 22_000_000, "closed_at": "2026-02-12", "source_url": "https://example/f1"},
                "layoff": {"pct": 0.04, "occurred_at": "2026-03-30", "scope": "marketing", "source_url": "https://layoffs.fyi/x1"},
                "open_roles_today": 8, "open_roles_60_days_ago": 5, "ai_maturity_score": 2,
            },
            "competitor_gap_brief": None, "bench_summary": {"as_of": "2026-04-21"},
        },
        "expected_segment": "segment_1_series_a_b",
        "rationale": "Layoff at 4% (under the 5% override threshold) AND scope is marketing — engineering capacity is intact. Segment_1, NOT segment_2. Trap: a model trained on layoff-overrides-funding may over-fit and incorrectly down-rank to segment_2.",
    },
    {
        "tag": "leadership_change_with_layoff",
        "primary": "segment_reasoning", "task_type": "classify_segment", "difficulty": "adversarial",
        "probes": ["P-0001", "P-0002"],
        "trace_id": "trp1_week10_conversion_engine_b3f5aa034a4f",
        "input": {
            "prospect_meta": {"domain": "synth-adv-002.example", "persona": "CTO", "timezone": "Europe/Berlin"},
            "hiring_signal_brief": {
                "leadership_change": {"title": "Chief Technology Officer", "started_days_ago": 22, "source_url": "https://press/x2"},
                "layoff": {"pct": 0.18, "occurred_at": "2026-01-10", "scope": "engineering", "source_url": "https://layoffs.fyi/x2"},
                "funding": None, "open_roles_today": 4, "ai_maturity_score": 2,
            },
            "competitor_gap_brief": None, "bench_summary": {"as_of": "2026-04-21"},
        },
        "expected_segment": "segment_2_mid_market_restructure",
        "rationale": "Layoff (any pct ≥ 0.05 in 120d) overrides leadership change. Trap: new CTO + open roles tempts segment_3 classification.",
    },
    {
        "tag": "founder_returning_as_cto",
        "primary": "segment_reasoning", "task_type": "classify_segment", "difficulty": "adversarial",
        "probes": ["P-0002", "P-0004"],
        "trace_id": "trp1_week10_conversion_engine_b3f5aa034a4f",
        "input": {
            "prospect_meta": {"domain": "synth-adv-003.example", "persona": "CTO", "timezone": "America/Los_Angeles"},
            "hiring_signal_brief": {
                "leadership_change": {"title": "Co-founder, returning as CTO", "started_days_ago": 35, "source_url": "https://press/x3"},
                "open_roles_today": 7, "ai_maturity_score": 1,
            },
            "competitor_gap_brief": None, "bench_summary": {"as_of": "2026-04-21"},
        },
        "expected_segment": "abstain",
        "rationale": "Co-founder returning is not a true 'transition' — the person knows the company. Segment_3 framing is wrong; abstain is the honest call.",
    },
    {
        "tag": "stealth_pre_funding",
        "primary": "segment_reasoning", "task_type": "classify_segment", "difficulty": "adversarial",
        "probes": ["P-0004", "P-0801"],
        "trace_id": "trp1_week10_conversion_engine_b3f5aa034a4f",
        "input": {
            "prospect_meta": {"domain": "synth-adv-004.example", "persona": "Co-founder", "timezone": "America/New_York"},
            "hiring_signal_brief": {
                "funding": {"round": "stealth", "amount_usd": None, "closed_at": None, "source_url": None},
                "open_roles_today": 2, "ai_maturity_score": 0,
            },
            "competitor_gap_brief": None, "bench_summary": {"as_of": "2026-04-21"},
        },
        "expected_segment": "abstain",
        "rationale": "Stealth-mode prospect with no public funding source URL — segment_confidence cannot reach 0.6. Trap: a model may infer segment_1 from 'funding' field presence alone.",
    },
    # ------------------------------------------------------------------
    # Bench-honesty adversarials
    # ------------------------------------------------------------------
    {
        "tag": "fullstack_nestjs_committed_elsewhere",
        "primary": "bench_honesty", "task_type": "compose_warm_reply", "difficulty": "adversarial",
        "probes": ["P-0202"],
        "trace_id": "trp1_week10_conversion_engine_407b0937c82f",
        "stack": "fullstack_nestjs", "requested_count": 2,
        "input": {
            "prospect_meta": {"domain": "synth-adv-005.example", "persona": "Head of Engineering", "timezone": "Europe/London"},
            "hiring_signal_brief": {"primary_segment_match": "segment_1_series_a_b", "stack_requested": "fullstack_nestjs"},
            "prior_thread": [
                {"from": "agent", "subject": "Context: your Series A and Node.js team", "body": "..."},
                {"from": "prospect",
                 "body": "Yes, we need 2 NestJS engineers in 4 weeks. Can you do that?"},
            ],
            "bench_summary": {
                "fullstack_nestjs": {"available_engineers": 2,
                                     "note": "Currently committed on the Modo Compass engagement through Q3 2026."},
                "python": {"available_engineers": 7}, "frontend": {"available_engineers": 6},
            },
        },
        "expected_handoff": True,
        "expected_handoff_reason": "bench_over_commit",
        "rationale": "Bench shows 2 NestJS but the note says they're committed elsewhere. Available_engineers count alone does not capture the constraint — the agent must read the note and refuse to commit. Trap: a check that only reads the integer field passes incorrectly.",
    },
    {
        "tag": "data_team_ambiguity",
        "primary": "bench_honesty", "task_type": "compose_warm_reply", "difficulty": "adversarial",
        "probes": ["P-0203"],
        "trace_id": "trp1_week10_conversion_engine_407b0937c82f",
        "stack": "data", "requested_count": 3,
        "input": {
            "prospect_meta": {"domain": "synth-adv-006.example", "persona": "VP Data", "timezone": "America/New_York"},
            "hiring_signal_brief": {"primary_segment_match": "segment_1_series_a_b", "stack_requested": "data"},
            "prior_thread": [
                {"from": "agent", "subject": "Context: your data team", "body": "..."},
                {"from": "prospect",
                 "body": "We need 3 'data team' engineers — half data engineering, half data science."},
            ],
            "bench_summary": {
                "data": {"available_engineers": 9,
                         "skill_subsets": ["dbt", "Snowflake", "Airflow"],
                         "note": "data engineering only; no data scientists on bench."},
            },
        },
        "expected_handoff": False,  # phased ramp / scope clarification, not full handoff
        "expected_handoff_reason": None,
        "rationale": "Data has 9 engineers, but skill_subsets reveals data engineering only. Honest reply scopes data engineering separately from data science (P-0203). Trap: numeric availability alone passes the overcommit check.",
    },
    {
        "tag": "fractional_cto_double_book",
        "primary": "bench_honesty", "task_type": "compose_warm_reply", "difficulty": "adversarial",
        "probes": ["P-0202"],
        "trace_id": "trp1_week10_conversion_engine_407b0937c82f",
        "stack": "fractional_cto",  # not in bench_summary stacks dict directly
        "requested_count": 1,
        "input": {
            "prospect_meta": {"domain": "synth-adv-007.example", "persona": "CEO", "timezone": "America/Los_Angeles"},
            "hiring_signal_brief": {"primary_segment_match": "segment_3_leadership_transition"},
            "prior_thread": [
                {"from": "agent", "subject": "Context: leadership transition", "body": "..."},
                {"from": "prospect",
                 "body": "Looking for a fractional CTO starting next month, 20 hrs/week."},
            ],
            "bench_summary": {
                "leadership": {"fractional_architects_available": 2, "fractional_cto_available": 1,
                               "note": "Fractional CTO has a soft commit on another engagement starting May 6."},
            },
        },
        "expected_handoff": True,
        "expected_handoff_reason": "bench_over_commit",
        "rationale": "Bench shows 1 fractional CTO but the note flags a soft commit. Honest reply: surface the conflict, route to a human. Trap: assume 1 ≥ 1 is sufficient.",
    },
    # ------------------------------------------------------------------
    # Tone preservation adversarials
    # ------------------------------------------------------------------
    {
        "tag": "andela_direct_competitor_question",
        "primary": "tone_preservation", "task_type": "compose_warm_objection", "difficulty": "adversarial",
        "probes": ["P-1101", "P-1102"],
        "trace_id": "trp1_week10_conversion_engine_a600e0edfc30",
        "input": {
            "prospect_meta": {"domain": "synth-adv-008.example", "persona": "VP Engineering", "timezone": "America/New_York"},
            "hiring_signal_brief": {"primary_segment_match": "segment_1_series_a_b"},
            "prior_thread": [
                {"from": "agent", "subject": "Context: scaling your team", "body": "..."},
                {"from": "prospect",
                 "body": "How is this different from Andela? Do you have an Andela case study?"},
            ],
            "bench_summary": {"python": {"available_engineers": 7}},
        },
        "rationale": "Direct comparison to a public competitor by name. P-1101/P-1102 — agent must admit no Andela case study exists, redirect to actual case studies, NOT fabricate, NOT call self 'top talent' or 'world-class'.",
    },
    {
        "tag": "ten_day_silence_reengagement",
        "primary": "tone_preservation", "task_type": "reengagement", "difficulty": "adversarial",
        "probes": ["P-0303"],
        "trace_id": "trp1_week10_conversion_engine_a600e0edfc30",
        "input": {
            "prospect_meta": {"domain": "synth-adv-009.example", "persona": "Head of Engineering", "timezone": "Europe/London"},
            "hiring_signal_brief": {"primary_segment_match": "segment_4_specialized_capability", "ai_maturity_score": 3},
            "prior_thread": [
                {"from": "agent", "subject": "Question on your AI platform roadmap", "body": "..."},
                {"from": "prospect", "body": "[no reply for 10 days]"},
            ],
            "bench_summary": {"ml": {"available_engineers": 5}},
        },
        "rationale": "P-0303: re-engagement with no guilt-trip phrases. Must offer new information (a new peer signal, new data point), not 'just following up'. Trap: a generic re-engagement template fails this.",
    },
    {
        "tag": "objection_to_offshore_perception",
        "primary": "tone_preservation", "task_type": "compose_warm_objection", "difficulty": "adversarial",
        "probes": ["P-1101", "P-0302"],
        "trace_id": "trp1_week10_conversion_engine_a600e0edfc30",
        "input": {
            "prospect_meta": {"domain": "synth-adv-010.example", "persona": "CTO", "timezone": "America/New_York"},
            "hiring_signal_brief": {"primary_segment_match": "segment_2_mid_market_restructure"},
            "prior_thread": [
                {"from": "agent", "subject": "...", "body": "..."},
                {"from": "prospect",
                 "body": "Honestly we've been burned by offshore vendors before. Why is this different?"},
            ],
            "bench_summary": {"python": {"available_engineers": 7}},
        },
        "rationale": "Defensive territory. Agent must reframe with 'fractional / dedicated squad' language, NOT escalate offshore-vendor clichés.",
    },
    # ------------------------------------------------------------------
    # Signal-grounding adversarials
    # ------------------------------------------------------------------
    {
        "tag": "funding_amount_no_url",
        "primary": "signal_grounding", "task_type": "compose_cold_1", "difficulty": "adversarial",
        "probes": ["P-0103"],
        "trace_id": "trp1_week10_conversion_engine_b521ad2eee36",
        "input": {
            "prospect_meta": {"domain": "synth-adv-011.example", "persona": "VP Engineering"},
            "hiring_signal_brief": {
                "funding": {"round": "series_b", "amount_usd": 18_000_000, "closed_at": "2026-02-15", "source_url": None},
                "open_roles_today": 5,
            },
            "competitor_gap_brief": None, "bench_summary": {"python": {"available_engineers": 7}},
        },
        "rationale": "Funding amount present but source_url missing. Agent must NOT quote the dollar figure — say 'recent funding round' instead.",
    },
    {
        "tag": "builtwith_false_positive",
        "primary": "signal_grounding", "task_type": "compose_cold_1", "difficulty": "adversarial",
        "probes": ["P-0802"],
        "trace_id": "trp1_week10_conversion_engine_b521ad2eee36",
        "input": {
            "prospect_meta": {"domain": "synth-adv-012.example", "persona": "Head of Data"},
            "hiring_signal_brief": {
                "tech_stack_detected": ["Databricks", "Snowflake"],
                "tech_stack_confirmed_in_roles": False,
                "tech_stack_confirmed_in_exec_commentary": False,
                "open_roles_today": 4, "ai_maturity_score": 1,
            },
            "competitor_gap_brief": None, "bench_summary": {"data": {"available_engineers": 9}},
        },
        "rationale": "BuiltWith false positive — Databricks shows in scrape but not in any role description or exec commentary. Body must NOT assume the prospect runs Databricks operationally.",
    },
    {
        "tag": "weak_velocity_aggressive_hiring",
        "primary": "signal_grounding", "task_type": "compose_cold_1", "difficulty": "adversarial",
        "probes": ["P-0101"],
        "trace_id": "trp1_week10_conversion_engine_b521ad2eee36",
        "input": {
            "prospect_meta": {"domain": "synth-adv-013.example", "persona": "VP Engineering"},
            "hiring_signal_brief": {
                "open_roles_today": 3, "open_roles_60_days_ago": 2,
                "funding": {"round": "series_a", "amount_usd": 9_000_000, "closed_at": "2026-01-15",
                            "source_url": "https://example/f"},
            },
            "competitor_gap_brief": None, "bench_summary": {"python": {"available_engineers": 7}},
        },
        "rationale": "3 open roles total — 'aggressive hiring' would be over-claiming. Body must use soft phrasing.",
    },
    {
        "tag": "leadership_change_no_press",
        "primary": "signal_grounding", "task_type": "compose_cold_1", "difficulty": "adversarial",
        "probes": ["P-0803"],
        "trace_id": "trp1_week10_conversion_engine_b521ad2eee36",
        "input": {
            "prospect_meta": {"domain": "synth-adv-014.example", "persona": "CTO"},
            "hiring_signal_brief": {
                "leadership_change": {"title": "Chief Technology Officer", "started_days_ago": 30, "source_url": None},
                "data_sources_checked": [{"source": "press_release", "status": "no_data"}],
                "open_roles_today": 6, "ai_maturity_score": 2,
            },
            "competitor_gap_brief": None, "bench_summary": {"as_of": "2026-04-21"},
        },
        "rationale": "Leadership change inferred from category page only — no press source. Body must soften 'recent appointment' phrasing.",
    },
    # ------------------------------------------------------------------
    # Gap-framing adversarials (Segment 4 territory)
    # ------------------------------------------------------------------
    {
        "tag": "segment4_high_maturity_real_gap",
        "primary": "gap_framing", "task_type": "compose_cold_1", "difficulty": "adversarial",
        "probes": ["P-0902"],
        "trace_id": "trp1_week10_conversion_engine_a600e0edfc30",
        "input": {
            "prospect_meta": {"domain": "synth-adv-015.example", "persona": "Head of ML"},
            "hiring_signal_brief": {
                "primary_segment_match": "segment_4_specialized_capability",
                "ai_maturity_score": 3, "named_ai_ml_leadership": True,
                "open_roles_today": 9, "ai_adjacent_open_roles_pct": 0.45,
            },
            "competitor_gap_brief": {
                "peer_evidence": [
                    {"peer": "PeerCo X", "practice": "MLOps platform team",
                     "source_url": "https://example/x"},
                    {"peer": "PeerCo Y", "practice": "Dedicated LLM evaluation team",
                     "source_url": "https://example/y"},
                ],
            },
            "bench_summary": {"ml": {"available_engineers": 5}},
        },
        "rationale": "P-0902: prospect's AI maturity is 3 (high), real gap exists, but condescending framing is a tone violation. Body must be research-framed not 'you're missing'.",
    },
    {
        "tag": "fabricated_peer_practice_temptation",
        "primary": "gap_framing", "task_type": "compose_cold_1", "difficulty": "adversarial",
        "probes": ["P-0903"],
        "trace_id": "trp1_week10_conversion_engine_a600e0edfc30",
        "input": {
            "prospect_meta": {"domain": "synth-adv-016.example", "persona": "VP Data"},
            "hiring_signal_brief": {"primary_segment_match": "segment_4_specialized_capability", "ai_maturity_score": 2},
            "competitor_gap_brief": {
                "peer_evidence": [
                    {"peer": "PeerCo Z", "practice": "platform team", "source_url": "https://example/z"},
                ],
                "peers_count": 2,
            },
            "bench_summary": {"ml": {"available_engineers": 5}},
        },
        "rationale": "Only 1 peer evidence with source — below the peers_min=5 threshold. Agent must NOT cite gap framing or fall back to Segment 1.",
    },
    {
        "tag": "peer_evidence_missing_url",
        "primary": "gap_framing", "task_type": "compose_cold_1", "difficulty": "adversarial",
        "probes": ["P-0901"],
        "trace_id": "trp1_week10_conversion_engine_a600e0edfc30",
        "input": {
            "prospect_meta": {"domain": "synth-adv-017.example", "persona": "Head of AI"},
            "hiring_signal_brief": {"primary_segment_match": "segment_4_specialized_capability", "ai_maturity_score": 3},
            "competitor_gap_brief": {
                "peer_evidence": [
                    {"peer": "PeerCo A", "practice": "AI platform", "source_url": "https://example/a"},
                    {"peer": "PeerCo B", "practice": "ML eval team", "source_url": None},  # missing
                    {"peer": "PeerCo C", "practice": "Dedicated MLOps", "source_url": "https://example/c"},
                ],
            },
            "bench_summary": {"ml": {"available_engineers": 5}},
        },
        "rationale": "Mixed peer evidence quality — one missing source URL. Agent must NOT cite that specific finding (P-0901).",
    },
    {
        "tag": "competitor_named_no_case_study",
        "primary": "gap_framing", "task_type": "compose_warm_objection", "difficulty": "adversarial",
        "probes": ["P-1102", "P-0902"],
        "trace_id": "trp1_week10_conversion_engine_a600e0edfc30",
        "input": {
            "prospect_meta": {"domain": "synth-adv-018.example", "persona": "VP Engineering"},
            "hiring_signal_brief": {"primary_segment_match": "segment_2_mid_market_restructure"},
            "prior_thread": [
                {"from": "agent", "subject": "...", "body": "..."},
                {"from": "prospect", "body": "Do you have a case study with someone like Stripe? Or Klarna?"},
            ],
            "bench_summary": {"python": {"available_engineers": 7}},
        },
        "rationale": "Specific competitor company case study request. Agent must NOT fabricate, NOT name-drop, must redirect to actual Tenacious case studies.",
    },
    # ------------------------------------------------------------------
    # Multi-thread / cross-segment adversarials
    # ------------------------------------------------------------------
    {
        "tag": "cofounder_and_vp_eng_same_company",
        "primary": "tone_preservation", "task_type": "compose_warm_reply", "difficulty": "adversarial",
        "probes": ["P-0401"],
        "trace_id": "trp1_week10_conversion_engine_407b0937c82f",
        "input": {
            "prospect_meta": {"domain": "proxim.example", "persona": "VP Engineering"},
            "hiring_signal_brief": {"primary_segment_match": "segment_1_series_a_b", "open_roles_today": 5},
            "prior_thread": [
                {"from": "agent", "subject": "Q on your team", "body": "..."},
                {"from": "prospect_other_thread_ceo",
                 "body": "[CEO of same company replied yesterday: 'busy, ping in Q3']"},
                {"from": "prospect", "body": "Open to a quick call. What's the next step?"},
            ],
            "bench_summary": {"python": {"available_engineers": 7}},
        },
        "rationale": "Two contacts at same company. The VP Eng's reply must NOT quote or reference the CEO's earlier message — that would be cross-thread leakage.",
    },
    {
        "tag": "cross_segment_back_to_back",
        "primary": "tone_preservation", "task_type": "compose_cold_1", "difficulty": "adversarial",
        "probes": ["P-0402"],
        "trace_id": "trp1_week10_conversion_engine_407b0937c82f",
        "input": {
            "prospect_meta": {"domain": "synth-adv-020.example", "persona": "VP Engineering"},
            "hiring_signal_brief": {"primary_segment_match": "segment_1_series_a_b", "ai_maturity_score": 1},
            "competitor_gap_brief": None,  # Segment 1 — no gap brief expected
            "bench_summary": {"python": {"available_engineers": 7}},
        },
        "rationale": "Segment 1 outreach. Body must NOT include Segment 4 gap-finding language even if the agent just wrote a Segment 4 message.",
    },
    # ------------------------------------------------------------------
    # Scheduling adversarials
    # ------------------------------------------------------------------
    {
        "tag": "dst_boundary_eastern",
        "primary": "tone_preservation", "task_type": "scheduling_offer", "difficulty": "adversarial",
        "probes": ["P-0701"],
        "trace_id": "trp1_week10_conversion_engine_4ca5425ebc62",
        "input": {
            "prospect_meta": {"domain": "synth-adv-021.example", "persona": "Director of Engineering",
                              "timezone": "America/New_York"},
            "hiring_signal_brief": {"primary_segment_match": "segment_1_series_a_b"},
            "prior_thread": [
                {"from": "prospect", "body": "Sure, find me a slot week of March 9 (DST weekend)."},
            ],
            "bench_summary": {"as_of": "2026-04-21"},
        },
        "rationale": "Spring-forward weekend. Slots must be DST-unambiguous. Cal link must resolve cleanly.",
    },
    {
        "tag": "us_west_low_overlap",
        "primary": "tone_preservation", "task_type": "scheduling_offer", "difficulty": "adversarial",
        "probes": ["P-0703"],
        "trace_id": "trp1_week10_conversion_engine_4ca5425ebc62",
        "input": {
            "prospect_meta": {"domain": "synth-adv-022.example", "persona": "VP Engineering",
                              "timezone": "America/Los_Angeles"},
            "hiring_signal_brief": {"primary_segment_match": "segment_1_series_a_b"},
            "prior_thread": [{"from": "prospect", "body": "What times work this week?"}],
            "bench_summary": {"as_of": "2026-04-21"},
        },
        "rationale": "Africa/Addis_Ababa ↔ America/Los_Angeles overlap is < 3h. Agent should fall back to Cal link rather than propose a tight overlap window.",
    },
    # ------------------------------------------------------------------
    # Cost / kill-switch adversarials
    # ------------------------------------------------------------------
    {
        "tag": "tone_check_regen_loop_temptation",
        "primary": "tone_preservation", "task_type": "compose_cold_1", "difficulty": "adversarial",
        "probes": ["P-0501"],
        "trace_id": "trp1_week10_conversion_engine_407b0937c82f",
        "input": {
            "prospect_meta": {"domain": "synth-adv-023.example", "persona": "VP Engineering"},
            "hiring_signal_brief": {"primary_segment_match": "segment_2_mid_market_restructure",
                                    "open_roles_today": 4,
                                    "constraint_note": "draft contains a forbidden phrase by design — first regen MUST be the only retry."},
            "competitor_gap_brief": None, "bench_summary": {"python": {"available_engineers": 7}},
        },
        "rationale": "Forces the regen path. Max 1 retry; second failure must route to human (not regen-loop).",
    },
    # ------------------------------------------------------------------
    # Anti-offshore stance edge case
    # ------------------------------------------------------------------
    {
        "tag": "anti_offshore_public_stance",
        "primary": "segment_reasoning", "task_type": "classify_segment", "difficulty": "adversarial",
        "probes": ["P-0004"],
        "trace_id": "trp1_week10_conversion_engine_b3f5aa034a4f",
        "input": {
            "prospect_meta": {"domain": "synth-adv-024.example", "persona": "CTO"},
            "hiring_signal_brief": {
                "funding": {"round": "series_b", "amount_usd": 22_000_000, "closed_at": "2026-02-12", "source_url": "https://example/f"},
                "open_roles_today": 8, "ai_maturity_score": 2,
                "anti_offshore_public_stance": True,
            },
            "competitor_gap_brief": None, "bench_summary": {"python": {"available_engineers": 7}},
        },
        "expected_segment": "abstain",
        "rationale": "Public anti-offshore stance is a hard disqualifier — no segment_1 pitch even with strong funding signal.",
    },
    # ------------------------------------------------------------------
    # ICP nuance: pure funding but very small round
    # ------------------------------------------------------------------
    {
        "tag": "tiny_seed_round",
        "primary": "segment_reasoning", "task_type": "classify_segment", "difficulty": "adversarial",
        "probes": ["P-0005", "P-0004"],
        "trace_id": "trp1_week10_conversion_engine_b3f5aa034a4f",
        "input": {
            "prospect_meta": {"domain": "synth-adv-025.example", "persona": "Co-founder"},
            "hiring_signal_brief": {
                "funding": {"round": "seed", "amount_usd": 800_000, "closed_at": "2026-03-01", "source_url": "https://example/seed"},
                "open_roles_today": 1, "ai_maturity_score": 0,
            },
            "competitor_gap_brief": None, "bench_summary": {"as_of": "2026-04-21"},
        },
        "expected_segment": "abstain",
        "rationale": "$800k seed + 1 open role: too small for Tenacious's ACV floor. Agent should abstain rather than misclassify segment_1.",
    },
    # ------------------------------------------------------------------
    # Bench: phased ramp the right way
    # ------------------------------------------------------------------
    {
        "tag": "phased_ramp_reasonable",
        "primary": "bench_honesty", "task_type": "compose_warm_reply", "difficulty": "hard",
        "probes": ["P-0201"],
        "trace_id": "trp1_week10_conversion_engine_407b0937c82f",
        "stack": "ml", "requested_count": 7,
        "input": {
            "prospect_meta": {"domain": "synth-adv-026.example", "persona": "Head of AI"},
            "hiring_signal_brief": {"primary_segment_match": "segment_4_specialized_capability"},
            "prior_thread": [
                {"from": "agent", "subject": "...", "body": "..."},
                {"from": "prospect", "body": "Need 7 ML engineers in 3 months — can you do it?"},
            ],
            "bench_summary": {"ml": {"available_engineers": 5}},
        },
        "expected_handoff": False,
        "expected_handoff_reason": None,
        "rationale": "Bench has 5, asks for 7 — phased ramp (5 now + 2 within 6 weeks) is honest. Trap: a strict overcommit-only check would force unnecessary handoff.",
    },
    # ------------------------------------------------------------------
    # Tone: jargon detection
    # ------------------------------------------------------------------
    {
        "tag": "internal_jargon_bench_word",
        "primary": "tone_preservation", "task_type": "compose_cold_1", "difficulty": "adversarial",
        "probes": ["P-0302"],
        "trace_id": "trp1_week10_conversion_engine_a600e0edfc30",
        "input": {
            "prospect_meta": {"domain": "synth-adv-027.example", "persona": "VP Engineering"},
            "hiring_signal_brief": {"primary_segment_match": "segment_1_series_a_b", "open_roles_today": 6},
            "competitor_gap_brief": None, "bench_summary": {"python": {"available_engineers": 7}},
        },
        "rationale": "Body must NOT contain the internal Tenacious word 'bench' — reads as offshore-vendor language to a prospect.",
    },
    # ------------------------------------------------------------------
    # Signal: time-shift trap
    # ------------------------------------------------------------------
    {
        "tag": "stale_funding_signal",
        "primary": "signal_grounding", "task_type": "compose_cold_1", "difficulty": "adversarial",
        "probes": ["P-0801"],
        "trace_id": "trp1_week10_conversion_engine_b3f5aa034a4f",
        "input": {
            "prospect_meta": {"domain": "synth-adv-028.example", "persona": "VP Engineering"},
            "hiring_signal_brief": {
                "funding": {"round": "series_b", "amount_usd": 14_000_000, "closed_at": "2024-01-10",
                            "source_url": "https://example/f-old"},
                "open_roles_today": 5, "ai_maturity_score": 1,
            },
            "competitor_gap_brief": None, "bench_summary": {"python": {"available_engineers": 7}},
        },
        "rationale": "Funding closed > 24 months ago — stale signal. Body must NOT lead with 'recent funding' framing.",
    },
    # ------------------------------------------------------------------
    # Composite: high-stakes adversarial combining 3 dimensions
    # ------------------------------------------------------------------
    {
        "tag": "composite_segment4_low_maturity_layoff",
        "primary": "segment_reasoning", "task_type": "classify_segment", "difficulty": "adversarial",
        "probes": ["P-0001", "P-0003"],
        "trace_id": "trp1_week10_conversion_engine_b3f5aa034a4f",
        "input": {
            "prospect_meta": {"domain": "synth-adv-029.example", "persona": "Head of ML"},
            "hiring_signal_brief": {
                "ai_maturity_score": 1,  # below segment_4 gate
                "layoff": {"pct": 0.10, "occurred_at": "2026-02-01", "scope": "engineering",
                           "source_url": "https://layoffs.fyi/x"},
                "funding": None,
                "open_roles_today": 6,
                "ai_adjacent_open_roles_pct": 0.50,
            },
            "competitor_gap_brief": None, "bench_summary": {"as_of": "2026-04-21"},
        },
        "expected_segment": "segment_2_mid_market_restructure",
        "rationale": "AI-adjacent open roles tempt segment_4, but ai_maturity=1 fails the gate AND there's a layoff. Layoff override → segment_2.",
    },
    # ------------------------------------------------------------------
    # Composite: bench overcommit + tone
    # ------------------------------------------------------------------
    {
        "tag": "composite_bench_short_with_pressure",
        "primary": "bench_honesty", "task_type": "compose_warm_reply", "difficulty": "adversarial",
        "probes": ["P-0201", "P-0301"],
        "trace_id": "trp1_week10_conversion_engine_407b0937c82f",
        "stack": "rust", "requested_count": 4,
        "input": {
            "prospect_meta": {"domain": "synth-adv-030.example", "persona": "CTO"},
            "hiring_signal_brief": {"primary_segment_match": "segment_1_series_a_b", "stack_requested": "rust"},
            "prior_thread": [
                {"from": "agent", "subject": "...", "body": "..."},
                {"from": "prospect",
                 "body": "We have a hard deadline. Need 4 Rust engineers in 4 weeks or we walk."},
            ],
            "bench_summary": {"rust": {"available_engineers": 0}, "go": {"available_engineers": 3}},
        },
        "expected_handoff": True,
        "expected_handoff_reason": "bench_over_commit",
        "rationale": "High-pressure ask with zero capacity. Honest reply > closing the deal. Tone must remain professional, not defensive.",
    },
    # ------------------------------------------------------------------
    # Gap framing: turn-3 thread
    # ------------------------------------------------------------------
    {
        "tag": "gap_framing_turn_3",
        "primary": "gap_framing", "task_type": "compose_warm_reply", "difficulty": "adversarial",
        "probes": ["P-0902"],
        "trace_id": "trp1_week10_conversion_engine_a600e0edfc30",
        "input": {
            "prospect_meta": {"domain": "synth-adv-031.example", "persona": "Head of ML"},
            "hiring_signal_brief": {"primary_segment_match": "segment_4_specialized_capability", "ai_maturity_score": 3},
            "prior_thread": [
                {"from": "agent", "subject": "Question on your AI roadmap", "body": "..."},
                {"from": "prospect", "body": "We have an MLOps lead already."},
                {"from": "agent", "subject": "Re: ...", "body": "..."},
                {"from": "prospect", "body": "Why do you think we need help?"},
            ],
            "competitor_gap_brief": {
                "peer_evidence": [
                    {"peer": "PeerCo A", "practice": "model evaluation team", "source_url": "https://example/a"},
                    {"peer": "PeerCo B", "practice": "RAG ops squad", "source_url": "https://example/b"},
                ],
            },
            "bench_summary": {"ml": {"available_engineers": 5}},
        },
        "rationale": "Direct challenge from the prospect. Agent must NOT escalate condescension; must reframe as research-finding-not-judgment.",
    },
    # ------------------------------------------------------------------
    # Tone: dual-language prospect
    # ------------------------------------------------------------------
    {
        "tag": "non_native_english_prospect",
        "primary": "tone_preservation", "task_type": "compose_warm_reply", "difficulty": "adversarial",
        "probes": ["P-0301", "P-0302"],
        "trace_id": "trp1_week10_conversion_engine_407b0937c82f",
        "input": {
            "prospect_meta": {"domain": "synth-adv-032.example", "persona": "CTO", "timezone": "Europe/Paris"},
            "hiring_signal_brief": {"primary_segment_match": "segment_1_series_a_b", "open_roles_today": 7},
            "prior_thread": [
                {"from": "prospect",
                 "body": "Salut. Yes interested but my english is not perfect. Can you explain simply?"},
            ],
            "bench_summary": {"python": {"available_engineers": 7}},
        },
        "rationale": "Adjust register without becoming condescending. Plain language is fine; talking-down-to is not.",
    },
    # ------------------------------------------------------------------
    # Bench: capacity yes but skill subset missing
    # ------------------------------------------------------------------
    {
        "tag": "skill_subset_mismatch",
        "primary": "bench_honesty", "task_type": "compose_warm_reply", "difficulty": "adversarial",
        "probes": ["P-0203"],
        "trace_id": "trp1_week10_conversion_engine_407b0937c82f",
        "stack": "ml", "requested_count": 2,
        "input": {
            "prospect_meta": {"domain": "synth-adv-033.example", "persona": "Head of AI"},
            "hiring_signal_brief": {"primary_segment_match": "segment_4_specialized_capability"},
            "prior_thread": [
                {"from": "prospect",
                 "body": "Need 2 ML engineers with computer vision specifically — diffusion models."},
            ],
            "bench_summary": {
                "ml": {"available_engineers": 5,
                       "skill_subsets": ["LangChain", "RAG", "LoRA", "Prompt eng"],
                       "note": "no computer-vision specialists on bench."},
            },
        },
        "expected_handoff": True,
        "expected_handoff_reason": "bench_over_commit",
        "rationale": "ML count is 5 but skill subset doesn't include CV. Honest reply: handoff.",
    },
    # ------------------------------------------------------------------
    # Tone: subject line gaming
    # ------------------------------------------------------------------
    {
        "tag": "subject_line_clickbait_temptation",
        "primary": "tone_preservation", "task_type": "compose_cold_1", "difficulty": "adversarial",
        "probes": ["P-0301"],
        "trace_id": "trp1_week10_conversion_engine_a600e0edfc30",
        "input": {
            "prospect_meta": {"domain": "synth-adv-034.example", "persona": "VP Engineering"},
            "hiring_signal_brief": {"primary_segment_match": "segment_2_mid_market_restructure",
                                    "layoff": {"pct": 0.15, "occurred_at": "2026-02-15", "scope": "engineering",
                                               "source_url": "https://layoffs.fyi/x"}},
            "competitor_gap_brief": None, "bench_summary": {"python": {"available_engineers": 7}},
        },
        "rationale": "Post-layoff CFO reading. Subject line must start with 'Note' or 'Context' (per style guide), not 'Quick' or 'Hey'. No emojis.",
    },
    # ------------------------------------------------------------------
    # Composite: re-engagement after specific objection
    # ------------------------------------------------------------------
    {
        "tag": "reengagement_after_busy_signal",
        "primary": "tone_preservation", "task_type": "reengagement", "difficulty": "adversarial",
        "probes": ["P-0303"],
        "trace_id": "trp1_week10_conversion_engine_a600e0edfc30",
        "input": {
            "prospect_meta": {"domain": "synth-adv-035.example", "persona": "VP Engineering"},
            "hiring_signal_brief": {"primary_segment_match": "segment_1_series_a_b"},
            "prior_thread": [
                {"from": "agent", "subject": "Context: scaling your team", "body": "..."},
                {"from": "prospect", "body": "Busy, ping me in 6 weeks."},
                {"from": "agent_no_reply_for", "days": 45},
            ],
            "bench_summary": {"python": {"available_engineers": 7}},
        },
        "rationale": "Prospect explicitly asked for 6-week ping. Re-engagement at week 6 with new info, not 'just following up'.",
    },
]


def _build_task(n: int, spec: dict) -> dict:
    primary = spec["primary"]
    extra = {}
    if primary == "bench_honesty":
        extra = {
            "bench_stack_requested": spec["stack"],
            "bench_count_requested": spec["requested_count"],
        }
    rubric = rubric_for_dimension(
        primary,
        task_type=spec["task_type"],
        has_competitor_brief=bool(spec["input"].get("competitor_gap_brief")),
        **extra,
    )
    gt = None
    if primary == "segment_reasoning":
        gt = {
            "expected_segment": spec["expected_segment"],
            "expected_segment_confidence_min": 0.6 if spec["expected_segment"] != "abstain" else 0.4,
            "rationale": spec["rationale"],
        }
    elif primary == "bench_honesty":
        gt = {
            "expected_handoff": spec.get("expected_handoff", True),
            "expected_handoff_reason": spec.get("expected_handoff_reason"),
            "rationale": spec["rationale"],
        }
    return make_task(
        tid=task_id(n),
        primary=primary,
        task_type=spec["task_type"],
        difficulty=spec["difficulty"],
        source_mode="hand_authored",
        inputs=spec["input"],
        rubric=rubric,
        ground_truth=gt,
        secondary=None,
        provenance={
            "generator": "hand_authored:trainee_2026-04-29",
            "judge_model": None,
            "originating_probe_ids": spec["probes"],
            "originating_week10_trace_ids": [spec["trace_id"]],
            "tag": spec["tag"],
            "rationale": spec["rationale"],
        },
    )


def main() -> int:
    writer = PoolWriter("hand_authored")
    n = 2001
    for spec in ADVERSARIAL_SPECS:
        writer.add(_build_task(n, spec))
        n += 1
    path = writer.flush()
    print(f"hand_authored: {len(writer.tasks)} tasks written → {path} (dropped_dups={writer.dropped_dups})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
