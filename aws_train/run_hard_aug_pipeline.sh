#!/usr/bin/env bash
# Error-targeted Hinglish pipeline. Run from repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-$ROOT/.venv_codemix/bin/python}"

echo "== 1. INT8 error dump (frozen val + train) =="
"$PY" -u aws_train/error_analysis_codemix.py

echo "== 2. Layer 1 clean (leak/dedup/dual-gate; Bedrock rescue+relabel unless SKIP_BEDROCK=1) =="
if [[ "${SKIP_BEDROCK:-0}" == "1" ]]; then
  "$PY" -u aws_train/layer1_clean_codemix.py --skip-bedrock
else
  "$PY" -u aws_train/layer1_clean_codemix.py "$@"
fi

echo "== 3. Seeded hard-aug =="
if [[ "${SKIP_BEDROCK:-0}" == "1" ]]; then
  echo "SKIP_BEDROCK=1 — not generating synth"
else
  "$PY" -u aws_train/synth_hard_aug_bedrock.py
fi

echo "== 4. Modal ablations =="
modal run aws_train/train_codemix_modal.py --all-ablations
