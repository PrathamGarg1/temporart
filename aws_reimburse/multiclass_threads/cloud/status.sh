#!/usr/bin/env bash
# Show progress of the EC2 training run (works from any machine with the AWS CLI).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="$HERE/run_state.json"
B=$(python3 -c "import json;print(json.load(open('$S'))['bucket'])")
I=$(python3 -c "import json;print(json.load(open('$S'))['instance_id'])")
R=$(python3 -c "import json;print(json.load(open('$S'))['region'])")
echo "instance: $(aws ec2 describe-instances --region $R --instance-ids $I --query 'Reservations[0].Instances[0].[State.Name,InstanceType,LaunchTime]' --output text)"
echo "--- run_meta"; aws s3 cp s3://$B/progress/run_meta.json - --region $R 2>/dev/null || echo "(not yet)"
echo; echo "--- last log lines"; aws s3 cp s3://$B/progress/job.log - --region $R 2>/dev/null | tr '\r' '\n' | grep -v "^$" | tail -${1:-12} || echo "(not yet)"
echo; echo "--- results"; aws s3 ls s3://$B/results/ --region $R 2>/dev/null || echo "(not yet)"
