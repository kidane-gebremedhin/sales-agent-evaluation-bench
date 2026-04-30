#!/bin/bash
# Act IV — γ sweep runner for SimPO training.
# Runs training + eval for each γ value sequentially.
#
# Usage:
#   bash training/gamma_sweep.sh            # full sweep
#   bash training/gamma_sweep.sh --dry-run  # config check only
#
# Designed for Colab T4 or any CUDA machine.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
fi

echo "================================================================="
echo "  SimPO γ Sweep — Tenacious Critic"
echo "  $(date -Iseconds)"
echo "================================================================="
echo ""

GAMMAS=(0.3 0.5 1.0 1.5)

# Phase 1: Training
echo "Phase 1: Training each γ..."
echo "---"

for GAMMA in "${GAMMAS[@]}"; do
    echo ""
    echo ">>> Training γ=$GAMMA"
    python training/train_simpo.py --gamma "$GAMMA" $DRY_RUN 2>&1 | tee "training/logs/train_gamma_${GAMMA}.log"
    echo "<<< Done γ=$GAMMA"
    echo ""
done

if [[ -n "$DRY_RUN" ]]; then
    echo "[DRY RUN] Sweep config validated. No training performed."
    exit 0
fi

# Phase 2: Evaluation
echo ""
echo "Phase 2: Evaluating all adapters on dev partition..."
echo "---"
python training/eval_dev.py --sweep 2>&1 | tee training/logs/eval_sweep.log

echo ""
echo "================================================================="
echo "  γ Sweep Complete"
echo "  Results: ablations/ablation_results.json"
echo "  Traces:  ablations/held_out_traces.jsonl"
echo "  $(date -Iseconds)"
echo "================================================================="
