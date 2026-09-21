#!/bin/bash
cd /Users/prathamgarg/Desktop/senior-sde-audit-surakshanet
echo "=== processes ==="
pgrep -fl 'codemix_convert|suraksha_codemix' || echo "(none)"
echo "=== screen ==="
screen -ls | grep -i suraksha || true
echo "=== counts ==="
for s in train val test; do
  m=codemix_hinglish/${s}_meta.jsonl
  a=codemix_hinglish/${s}.csv
  q=codemix_hinglish/${s}_quarantine.csv
  mn=0; an=0; qn=0
  [[ -f $m ]] && mn=$(wc -l < $m | tr -d ' ')
  [[ -f $a ]] && an=$(($(wc -l < $a | tr -d ' ')-1))
  [[ -f $q ]] && qn=$(($(wc -l < $q | tr -d ' ')-1))
  case $s in train) t=20183;; *) t=6728;; esac
  pct=$(python3 -c "print(f'{100*$mn/$t:.1f}')" 2>/dev/null || echo '?')
  echo "$s: meta=$mn/$t ($pct%) accepted~$an quarantine~$qn"
done
echo "=== log tail ==="
tail -8 codemix_hinglish/full_convert.log 2>/dev/null
