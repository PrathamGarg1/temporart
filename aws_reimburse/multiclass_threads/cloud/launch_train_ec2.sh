#!/usr/bin/env bash
# Launch an isolated GPU training run on EC2 (us-east-1). Safe to close the laptop.
#
#   bash cloud/launch_train_ec2.sh
#
# Isolation / evidence:
#   * dedicated S3 bucket, IAM role (this bucket only), SG with NO inbound rules, no SSH key
#   * every resource tagged Project=surakshanet-multiclass-threads, Job=multiclass_train_v2_gpu
#   * instance self-terminates when done, or after a 5 h hard cap
#   * progress log synced to S3 every 60 s; results land in s3://BUCKET/results/
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
REGION=us-east-1
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
STAMP=$(date -u +%Y%m%d%H%M%S)
BUCKET="suraksha-mc-train-${ACCOUNT}-${STAMP}"
ROLE="suraksha-mc-train-role-${STAMP}"
PROFILE="suraksha-mc-train-profile-${STAMP}"
SG_NAME="suraksha-mc-train-sg-${STAMP}"
TAGS_KV="Key=Project,Value=surakshanet-multiclass-threads"
JOB_TAG="Key=Job,Value=multiclass_train_v2_gpu"
STATE="$HERE/cloud/run_state.json"

echo "bucket s3://$BUCKET"
aws s3 mb "s3://$BUCKET" --region $REGION >/dev/null
aws s3api put-bucket-tagging --bucket "$BUCKET" --tagging "TagSet=[{Key=Project,Value=surakshanet-multiclass-threads},{Key=Job,Value=multiclass_train_v2_gpu}]"

echo "packaging"
STAGE=$(mktemp -d)
mkdir -p "$STAGE/multiclass_threads/scripts" "$STAGE/multiclass_threads/dataset/all"
cp "$HERE"/scripts/{train_v2.py,benchmark_v2.py,cost_ledger.py} "$STAGE/multiclass_threads/scripts/"
cp "$HERE"/dataset/all/{train,val,test}.jsonl "$STAGE/multiclass_threads/dataset/all/"
tar -C "$STAGE" -czf /tmp/mc-job-$STAMP.tgz .
aws s3 cp /tmp/mc-job-$STAMP.tgz "s3://$BUCKET/job.tgz" --region $REGION --only-show-errors
rm -rf "$STAGE" /tmp/mc-job-$STAMP.tgz

echo "iam"
TRUST='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam create-role --role-name "$ROLE" --assume-role-policy-document "$TRUST" \
  --tags "$TAGS_KV" "$JOB_TAG" >/dev/null
aws iam put-role-policy --role-name "$ROLE" --policy-name s3-this-bucket --policy-document "{
 \"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",
 \"Action\":[\"s3:GetObject\",\"s3:PutObject\",\"s3:ListBucket\"],
 \"Resource\":[\"arn:aws:s3:::$BUCKET\",\"arn:aws:s3:::$BUCKET/*\"]}]}"
aws iam create-instance-profile --instance-profile-name "$PROFILE" >/dev/null
aws iam add-role-to-instance-profile --instance-profile-name "$PROFILE" --role-name "$ROLE"

VPC=$(aws ec2 describe-vpcs --region $REGION --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
SG=$(aws ec2 create-security-group --region $REGION --group-name "$SG_NAME" \
  --description "mc train: no inbound" --vpc-id "$VPC" \
  --tag-specifications "ResourceType=security-group,Tags=[{Key=Project,Value=surakshanet-multiclass-threads}]" \
  --query GroupId --output text)

AMI=$(aws ssm get-parameter --region $REGION \
  --name /aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.7-amazon-linux-2023/latest/ami-id \
  --query Parameter.Value --output text)
echo "ami $AMI"

cat > /tmp/mc-userdata-$STAMP.sh <<EOF
#!/bin/bash
export HOME=/root
BUCKET=$BUCKET
REGION=$REGION
EOF
cat >> /tmp/mc-userdata-$STAMP.sh <<'EOF'
shutdown -h +300 "hard cap 5h"   # terminate-on-shutdown => bills stop
exec > >(tee -a /var/log/job.log) 2>&1
W=/opt/job; mkdir -p $W
TOKEN=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
IID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
ITYPE=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-type)
echo "{\"instance_id\":\"$IID\",\"instance_type\":\"$ITYPE\",\"start_epoch\":$(date +%s),\"status\":\"starting\"}" > $W/run_meta.json

sync_loop() { while true; do
  aws s3 cp /var/log/job.log s3://$BUCKET/progress/job.log --region $REGION --only-show-errors
  aws s3 cp $W/run_meta.json s3://$BUCKET/progress/run_meta.json --region $REGION --only-show-errors
  for f in /opt/job/multiclass_threads/artifacts_v2/runs_*/trainer_state.json; do
    [ -f "$f" ] && aws s3 cp "$f" s3://$BUCKET/progress/$(basename $(dirname $f))_trainer_state.json --region $REGION --only-show-errors
  done; sleep 60; done; }
sync_loop &

set_status() { python3 - "$1" <<'PY'
import json,sys,time
p="/opt/job/run_meta.json"; d=json.load(open(p)); d["status"]=sys.argv[1]; d["updated_epoch"]=int(time.time()); json.dump(d,open(p,"w"))
PY
}
finish() {
  python3 - <<'PY'
import json,time
p="/opt/job/run_meta.json"; d=json.load(open(p)); d["end_epoch"]=int(time.time()); json.dump(d,open(p,"w"))
PY
  aws s3 cp /var/log/job.log s3://$BUCKET/progress/job.log --region $REGION --only-show-errors
  aws s3 cp $W/run_meta.json s3://$BUCKET/progress/run_meta.json --region $REGION --only-show-errors
  shutdown -h now
}
trap finish EXIT

set -x
aws s3 cp s3://$BUCKET/job.tgz /tmp/job.tgz --region $REGION --only-show-errors
tar -C $W -xzf /tmp/job.tgz
[ -f /opt/pytorch/bin/activate ] && source /opt/pytorch/bin/activate
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
nvidia-smi -L
pip install -q "transformers==4.57.6" accelerate numpy
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 3)" || { set_status "no_cuda"; exit 3; }

cd $W/multiclass_threads/scripts
export NO_LEDGER=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
set_status "training_isolated"
python -u train_v2.py --mode isolated --epochs 2 --batch 32 --max-len 256 --val-n 4000 --eval-batch 128 --workers 2
set_status "training_thread"
python -u train_v2.py --mode thread --epochs 2 --batch 32 --max-len 256 --val-n 4000 --eval-batch 128 --workers 2
set_status "benchmark"
BENCH_BS=128 python -u benchmark_v2.py
set_status "uploading"
A=$W/multiclass_threads/artifacts_v2
aws s3 cp $A/benchmark_report.json s3://$BUCKET/results/benchmark_report.json --region $REGION --only-show-errors
for m in isolated thread; do
  aws s3 cp $A/runs_$m/trainer_state.json s3://$BUCKET/results/trainer_state_$m.json --region $REGION --only-show-errors 2>/dev/null || \
  aws s3 cp $(ls -d $A/runs_$m/checkpoint-*/ | tail -1)trainer_state.json s3://$BUCKET/results/trainer_state_$m.json --region $REGION --only-show-errors || true
  aws s3 cp --recursive $A/model_$m s3://$BUCKET/results/model_$m --region $REGION --only-show-errors
done
set_status "done"
EOF

IID=""
for T in g5.xlarge g6.xlarge g4dn.xlarge; do
  echo "trying $T"
  if IID=$(aws ec2 run-instances --region $REGION --image-id "$AMI" --instance-type "$T" \
      --iam-instance-profile "Name=$PROFILE" --security-group-ids "$SG" \
      --user-data "file:///tmp/mc-userdata-$STAMP.sh" \
      --instance-initiated-shutdown-behavior terminate \
      --metadata-options HttpTokens=required,HttpEndpoint=enabled \
      --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":100,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
      --tag-specifications \
        "ResourceType=instance,Tags=[{Key=Name,Value=suraksha-mc-train-$STAMP},{Key=Project,Value=surakshanet-multiclass-threads},{Key=Job,Value=multiclass_train_v2_gpu}]" \
        "ResourceType=volume,Tags=[{Key=Project,Value=surakshanet-multiclass-threads},{Key=Job,Value=multiclass_train_v2_gpu}]" \
      --query 'Instances[0].InstanceId' --output text 2>/tmp/mc-launch-err.txt); then
    ITYPE=$T; break
  else
    cat /tmp/mc-launch-err.txt | tail -2; IID=""
  fi
done
[ -n "$IID" ] || { echo "no GPU capacity/quota for any type"; exit 1; }
rm -f /tmp/mc-userdata-$STAMP.sh

LAUNCH=$(aws ec2 describe-instances --region $REGION --instance-ids "$IID" --query 'Reservations[0].Instances[0].LaunchTime' --output text)
cat > "$STATE" <<JSON
{"region":"$REGION","account":"$ACCOUNT","bucket":"$BUCKET","instance_id":"$IID","instance_type":"$ITYPE",
 "launch_time":"$LAUNCH","role":"$ROLE","profile":"$PROFILE","sg":"$SG","ami":"$AMI","ebs_gb":100,
 "tags":{"Project":"surakshanet-multiclass-threads","Job":"multiclass_train_v2_gpu"}}
JSON
echo "launched $IID ($ITYPE) at $LAUNCH"
echo "state: $STATE"
echo "check:  bash cloud/status.sh"
