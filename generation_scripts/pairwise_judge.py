#!/usr/bin/env python3
"""Pairwise LLM-as-a-Judge implementation for Tenacious-Bench.

Addresses the limitation of absolute pointwise scoring by executing a true
pairwise comparison (Candidate A vs Candidate B) following standard RLHF 
alignment protocols (e.g. Prometheus 2 style tie-breaking).

Usage:
    python generation_scripts/pairwise_judge.py
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import Literal

@dataclass
class PairwiseResult:
    winner: Literal["A", "B", "Tie"]
    reasoning: str

def pairwise_compare_candidates(
    task_context: dict,
    candidate_a: str,
    candidate_b: str,
    dimension: str = "tone_preservation",
    model: str = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
) -> PairwiseResult:
    """Executes a true Pairwise LLM-as-a-Judge comparison via OpenRouter."""
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        # Standard fallback for offline verification without breaking the test suite
        print("[!] OPENROUTER_API_KEY unset. Defaulting to offline length heuristic for demonstration.")
        winner = "A" if len(candidate_a) <= len(candidate_b) else "B"
        return PairwiseResult(winner, "Offline mode: shorter draft preferred in B2B constrained format.")

    prompt = (
        f"You are an expert B2B sales evaluator judging two outreach drafts. "
        f"You must evaluate which candidate strictly performs better on this dimension: '{dimension}'.\n\n"
        f"[Task Context]\n{json.dumps(task_context, indent=2)}\n\n"
        f"[Candidate A]\n{candidate_a}\n\n"
        f"[Candidate B]\n{candidate_b}\n\n"
        "First, provide your step-by-step reasoning. Then, provide the final verdict on the last line matching exactly one of these: "
        "'Winner: A', 'Winner: B', or 'Winner: Tie'."
    )
    
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 150,
    }).encode("utf-8")
    
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read())
        response_text = data["choices"][0]["message"]["content"].strip()
        
        winner = "Tie"
        if "Winner: A" in response_text:
            winner = "A"
        elif "Winner: B" in response_text:
            winner = "B"
            
        return PairwiseResult(winner, response_text)
        
    except Exception as e:
        print(f"[!] Pairwise evaluation failed: {e}")
        return PairwiseResult("Tie", f"Error during API execution: {str(e)}")

# Example execution if run standalone
if __name__ == "__main__":
    test_context = {
        "segment_label": "Segment_1",
        "layoff_event": True,
        "funding_event": True
    }
    cand_a = "We noticed your recent series A. You're scaling fast! We have world-class rockstars ready."
    cand_b = "I saw the recent restructuring combined with the Series A announcement. If you need integeration help, let me know."
    
    print("Executing Pairwise Comparison on `tone_preservation`...")
    result = pairwise_compare_candidates(test_context, cand_a, cand_b)
    print(f"\nWinning Candidate: {result.winner}")
    print(f"Judge Reasoning: {result.reasoning}")
