#!/usr/bin/env python3
"""Deploy quarantine fix to EC2 via SSM."""
import json
import subprocess
import time
from pathlib import Path

ROOT = Path("/Users/prathamgarg/Desktop/senior-sde-audit-surakshanet")
env = (ROOT / "aws_train/codemix_ec2.env").read_text()
kv = dict(
    line.split("=", 1) for line in env.splitlines() if "=" in line and not line.startswith("#")
)
bucket = kv["BUCKET"]
iid = kv["INSTANCE_ID"]
region = kv.get("REGION", "us-east-1")

loop = f"""#!/bin/bash
export AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1
cd /opt/suraksha
LOG=codemix_hinglish/full_convert.log
PY=/opt/suraksha/.venv/bin/python
BUCKET={bucket}
echo $$ > codemix_hinglish/full_convert.pid
echo "[$(date -Iseconds)] QWEN+MANTLE fix loop workers=2" >> $LOG
sync_out(){{ aws s3 sync codemix_hinglish s3://$BUCKET/codemix_hinglish --region us-east-1 --quiet || true; }}
while true; do
  for split in train val test; do
    case $split in train) t=20183;; *) t=6728;; esac
    a=0; q=0
    [[ -f codemix_hinglish/${{split}}.csv ]] && a=$(($(wc -l < codemix_hinglish/${{split}}.csv)-1))
    [[ -f codemix_hinglish/${{split}}_quarantine.csv ]] && q=$(($(wc -l < codemix_hinglish/${{split}}_quarantine.csv)-1))
    [[ $((a+q)) -ge $t ]] && continue
    echo "[$(date -Iseconds)] chunk $split a=$a q=$q" >> $LOG
    $PY -u aws_train/codemix_convert_bedrock.py --full --splits $split \\
      --backend mantle --model qwen.qwen3-32b \\
      --workers 2 --sleep 0.2 --max-retries 3 --limit 150 >> $LOG 2>&1 || sleep 30
    sync_out
  done
  ta=0;tq=0;va=0;vq=0;sa=0;sq=0
  [[ -f codemix_hinglish/train.csv ]] && ta=$(($(wc -l < codemix_hinglish/train.csv)-1))
  [[ -f codemix_hinglish/train_quarantine.csv ]] && tq=$(($(wc -l < codemix_hinglish/train_quarantine.csv)-1))
  [[ -f codemix_hinglish/val.csv ]] && va=$(($(wc -l < codemix_hinglish/val.csv)-1))
  [[ -f codemix_hinglish/val_quarantine.csv ]] && vq=$(($(wc -l < codemix_hinglish/val_quarantine.csv)-1))
  [[ -f codemix_hinglish/test.csv ]] && sa=$(($(wc -l < codemix_hinglish/test.csv)-1))
  [[ -f codemix_hinglish/test_quarantine.csv ]] && sq=$(($(wc -l < codemix_hinglish/test_quarantine.csv)-1))
  echo "[$(date -Iseconds)] progress train=$((ta+tq)) val=$((va+vq)) test=$((sa+sq))" >> $LOG
  if [[ $((ta+tq)) -ge 20183 && $((va+vq)) -ge 6728 && $((sa+sq)) -ge 6728 ]]; then
    echo done > codemix_hinglish/DONE; sync_out; break
  fi
  sleep 3
done
"""

clean_py = r"""
import pandas as pd
from pathlib import Path
for s in ['train','val','test']:
  q=Path(f'codemix_hinglish/{s}_quarantine.csv')
  if not q.exists():
    continue
  df=pd.read_csv(q)
  before=len(df)
  fr=df['fail_reason'].astype(str)
  keep = (~fr.str.contains('Throttl', na=False)
          & ~fr.str.startswith('api_error')
          & (fr != 'abuse_score_drop'))
  df2=df[keep]
  df2.to_csv(q, index=False)
  print(s, before, '->', len(df2), 'released', before-len(df2))
"""

cmds = [
    "cd /opt/suraksha",
    "pkill -f codemix_convert || true",
    "pkill -f suraksha_codemix_loop || true",
    "sleep 3",
    f"aws s3 cp s3://{bucket}/codemix_convert_bedrock.py aws_train/codemix_convert_bedrock.py --region us-east-1",
    "/opt/suraksha/.venv/bin/python - <<'PY'\n" + clean_py + "\nPY",
    "cat > aws_train/suraksha_codemix_loop.sh << 'LOOPEOF'\n" + loop + "LOOPEOF",
    "chmod +x aws_train/suraksha_codemix_loop.sh",
    "nohup bash aws_train/suraksha_codemix_loop.sh > codemix_hinglish/ec2_supervisor.out 2>&1 &",
    "sleep 25",
    "pgrep -fl codemix || echo DEAD",
    "wc -l codemix_hinglish/*_quarantine.csv",
    "tail -25 codemix_hinglish/full_convert.log",
]

params_path = Path("/tmp/ssm_fix_params.json")
params_path.write_text(json.dumps({"commands": cmds}))

cmd_id = subprocess.check_output(
    [
        "aws",
        "ssm",
        "send-command",
        "--region",
        region,
        "--instance-ids",
        iid,
        "--document-name",
        "AWS-RunShellScript",
        "--parameters",
        f"file://{params_path}",
        "--query",
        "Command.CommandId",
        "--output",
        "text",
    ],
    text=True,
).strip()
print("CMD", cmd_id)

for _ in range(20):
    time.sleep(5)
    out = subprocess.check_output(
        [
            "aws",
            "ssm",
            "get-command-invocation",
            "--region",
            region,
            "--command-id",
            cmd_id,
            "--instance-id",
            iid,
            "--output",
            "json",
        ],
        text=True,
    )
    data = json.loads(out)
    if data["Status"] not in ("Pending", "InProgress", "Delayed"):
        print("STATUS", data["Status"])
        print(data.get("StandardOutputContent", "")[-3000:])
        if data.get("StandardErrorContent"):
            print("ERR", data["StandardErrorContent"][-1500:])
        break
else:
    print("timeout waiting for SSM")
