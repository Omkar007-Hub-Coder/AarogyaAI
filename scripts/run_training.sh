#!/usr/bin/env bash
# scripts/run_training.sh
# Launches the EfficientNetB0 training pipeline with the TF venv.
# Usage: bash scripts/run_training.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV="$PROJECT_DIR/.venv-tf"
LOGFILE="$PROJECT_DIR/models/artifacts/training.log"

mkdir -p "$PROJECT_DIR/models/artifacts"

echo "=== AarogyaAI Training Run ===" | tee "$LOGFILE"
echo "Started: $(date)"                | tee -a "$LOGFILE"
echo "Python: $("$VENV/bin/python" --version 2>&1)" | tee -a "$LOGFILE"

cd "$PROJECT_DIR"

TF_CPP_MIN_LOG_LEVEL=3 \
PYTHONUNBUFFERED=1 \
"$VENV/bin/python" -m ml.training.train_yoga_classifier \
    --epochs-p1 15 \
    --epochs-p2 15 \
    2>&1 | tee -a "$LOGFILE"

echo "Finished: $(date)" | tee -a "$LOGFILE"
