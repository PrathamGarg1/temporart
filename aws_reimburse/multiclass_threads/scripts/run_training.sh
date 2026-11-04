#!/bin/zsh
# Resumable: re-running continues from the latest checkpoint; skips finished models.
export HF_HUB_OFFLINE=1
cd "$(dirname "$0")"
P=../../../aws_train/.venv/bin/python
L=../logs
[ -f ../artifacts_v2/model_isolated/config.json ] || $P -u train_v2.py --mode isolated --epochs 2 --val-n 4000 --resume >> $L/train_isolated.log 2>&1
[ -f ../artifacts_v2/model_thread/config.json ]   || $P -u train_v2.py --mode thread   --epochs 2 --val-n 3000 --batch 16 --max-len 192 --lr 3e-5 --resume >> $L/train_thread.log 2>&1
$P -u benchmark_v2.py > $L/benchmark_v2.log 2>&1
