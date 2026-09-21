#!/bin/bash
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
export AWS_REGION=us-east-1
export HOME="${HOME:-/Users/prathamgarg}"
cd /Users/prathamgarg/Desktop/senior-sde-audit-surakshanet || exit 1
mkdir -p codemix_hinglish
LOG=codemix_hinglish/full_convert.log
PY=.venv_codemix/bin/python
echo $$ > codemix_hinglish/full_convert.pid
echo "[$(date -Iseconds)] loop start pid=$$ workers=4 limit=400" >> "$LOG"

while true; do
  for split in train val test; do
    case $split in train) t=20183;; *) t=6728;; esac
    n=0
    [[ -f "codemix_hinglish/${split}_meta.jsonl" ]] && n=$(wc -l < "codemix_hinglish/${split}_meta.jsonl" | tr -d ' ')
    [[ "$n" -ge "$t" ]] && continue
    echo "[$(date -Iseconds)] chunk $split n=$n/$t" >> "$LOG"
    "$PY" -u aws_train/codemix_convert_bedrock.py --full --splits "$split" \
      --workers 4 --sleep 0.02 --max-retries 2 --limit 400 >> "$LOG" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "[$(date -Iseconds)] chunk exit=$rc; sleep 20" >> "$LOG"
      sleep 20
    fi
  done
  nt=$(wc -l < codemix_hinglish/train_meta.jsonl 2>/dev/null | tr -d ' '); nt=${nt:-0}
  nv=$(wc -l < codemix_hinglish/val_meta.jsonl 2>/dev/null | tr -d ' '); nv=${nv:-0}
  ns=$(wc -l < codemix_hinglish/test_meta.jsonl 2>/dev/null | tr -d ' '); ns=${ns:-0}
  echo "[$(date -Iseconds)] progress train=$nt val=$nv test=$ns" >> "$LOG"
  if [[ "$nt" -ge 20183 && "$nv" -ge 6728 && "$ns" -ge 6728 ]]; then
    echo "[$(date -Iseconds)] ALL DONE" >> "$LOG"
    # write summary
    "$PY" - <<'PY' >> "$LOG" 2>&1 || true
import json, pandas as pd
from pathlib import Path
root = Path("codemix_hinglish")
summary = {}
for split in ("train","val","test"):
    a = root / f"{split}.csv"
    q = root / f"{split}_quarantine.csv"
    if not a.exists():
        continue
    adf = pd.read_csv(a)
    qdf = pd.read_csv(q) if q.exists() else None
    ab_a = int((adf.label==0).sum())
    ab_q = int((qdf.label==0).sum()) if qdf is not None and len(qdf) else 0
    summary[split] = {
        "accepted": len(adf),
        "quarantined": 0 if qdf is None else len(qdf),
        "abusive_accept_rate": ab_a/(ab_a+ab_q) if (ab_a+ab_q) else None,
    }
(root/"conversion_report.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
    break
  fi
  sleep 2
done
rm -f codemix_hinglish/full_convert.pid
