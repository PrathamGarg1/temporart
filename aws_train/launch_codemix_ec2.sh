#!/usr/bin/env bash
# Launch EC2 conversion worker in us-east-1 (survives laptop sleep).
# Usage: bash aws_train/launch_codemix_ec2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
REGION=us-east-1
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
STAMP=$(date +%Y%m%d%H%M%S)
BUCKET="suraksha-codemix-${ACCOUNT}-${STAMP}"
KEY_NAME="suraksha-codemix-${STAMP}"
ROLE_NAME="suraksha-codemix-role-${STAMP}"
PROFILE_NAME="suraksha-codemix-profile-${STAMP}"
SG_NAME="suraksha-codemix-sg-${STAMP}"
INSTANCE_NAME="suraksha-codemix-${STAMP}"
STATE="$ROOT/aws_train/codemix_ec2.env"

echo "Creating bucket s3://$BUCKET"
aws s3 mb "s3://$BUCKET" --region "$REGION"

echo "Packaging job..."
STAGE=$(mktemp -d)
mkdir -p "$STAGE/aws_train" "$STAGE/codemix_hinglish" "$STAGE/assets/models" "$STAGE/aws_train/macd_hindi_cache"
cp aws_train/codemix_convert_bedrock.py aws_train/suraksha_codemix_loop.sh aws_train/requirements_codemix.txt "$STAGE/aws_train/"
# progress so EC2 resumes
cp -R codemix_hinglish/* "$STAGE/codemix_hinglish/" 2>/dev/null || true
cp -R assets/models/custom-macd-model "$STAGE/assets/models/"
cp -R aws_train/macd_hindi_cache/* "$STAGE/aws_train/macd_hindi_cache/" 2>/dev/null || true
# EC2 loop: sync to S3 periodically + finish flag
cat > "$STAGE/aws_train/suraksha_codemix_loop.sh" <<'EOF'
#!/bin/bash
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
export AWS_REGION=us-east-1
export AWS_DEFAULT_REGION=us-east-1
cd /opt/suraksha || exit 1
LOG=codemix_hinglish/full_convert.log
PY=/opt/suraksha/.venv/bin/python
BUCKET_FILE=/opt/suraksha/BUCKET
BUCKET=$(cat "$BUCKET_FILE")
echo $$ > codemix_hinglish/full_convert.pid
echo "[$(date -Iseconds)] EC2 loop start pid=$$" >> "$LOG"

sync_out() {
  aws s3 sync codemix_hinglish "s3://$BUCKET/codemix_hinglish" --region us-east-1 --quiet || true
}

while true; do
  for split in train val test; do
    case $split in train) t=20183;; *) t=6728;; esac
    n=0
    [[ -f "codemix_hinglish/${split}_meta.jsonl" ]] && n=$(wc -l < "codemix_hinglish/${split}_meta.jsonl" | tr -d ' ')
    [[ "$n" -ge "$t" ]] && continue
    echo "[$(date -Iseconds)] chunk $split n=$n/$t" >> "$LOG"
    "$PY" -u aws_train/codemix_convert_bedrock.py --full --splits "$split" \
      --workers 6 --sleep 0.02 --max-retries 2 --limit 500 >> "$LOG" 2>&1 || sleep 20
    sync_out
  done
  nt=$(wc -l < codemix_hinglish/train_meta.jsonl 2>/dev/null | tr -d ' '); nt=${nt:-0}
  nv=$(wc -l < codemix_hinglish/val_meta.jsonl 2>/dev/null | tr -d ' '); nv=${nv:-0}
  ns=$(wc -l < codemix_hinglish/test_meta.jsonl 2>/dev/null | tr -d ' '); ns=${ns:-0}
  echo "[$(date -Iseconds)] progress train=$nt val=$nv test=$ns" >> "$LOG"
  sync_out
  if [[ "$nt" -ge 20183 && "$nv" -ge 6728 && "$ns" -ge 6728 ]]; then
    echo "[$(date -Iseconds)] ALL DONE" >> "$LOG"
    echo done > codemix_hinglish/DONE
    sync_out
    break
  fi
  sleep 2
done
rm -f codemix_hinglish/full_convert.pid
EOF
chmod +x "$STAGE/aws_train/suraksha_codemix_loop.sh"
echo "$BUCKET" > "$STAGE/BUCKET"
tar -C "$STAGE" -czf /tmp/suraksha-codemix-job.tgz .
aws s3 cp /tmp/suraksha-codemix-job.tgz "s3://$BUCKET/job.tgz" --region "$REGION"
rm -rf "$STAGE"

echo "IAM role..."
TRUST='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document "$TRUST" >/dev/null
aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name codemix-inline --policy-document "{
  \"Version\": \"2012-10-17\",
  \"Statement\": [
    {\"Effect\":\"Allow\",\"Action\":[\"bedrock:*\",\"aws-marketplace:ViewSubscriptions\",\"aws-marketplace:Subscribe\"],\"Resource\":\"*\"},
    {\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\",\"s3:ListBucket\",\"s3:DeleteObject\"],\"Resource\":[\"arn:aws:s3:::$BUCKET\",\"arn:aws:s3:::$BUCKET/*\"]}
  ]
}"
aws iam create-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null
aws iam add-role-to-instance-profile --instance-profile-name "$PROFILE_NAME" --role-name "$ROLE_NAME"
echo "Waiting for instance profile propagation..."
sleep 15

echo "Key + SG..."
aws ec2 create-key-pair --region "$REGION" --key-name "$KEY_NAME" --query KeyMaterial --output text > "/tmp/${KEY_NAME}.pem"
chmod 600 "/tmp/${KEY_NAME}.pem"
VPCs=$(aws ec2 describe-vpcs --region "$REGION" --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
SG_ID=$(aws ec2 create-security-group --region "$REGION" --group-name "$SG_NAME" --description "codemix ssh" --vpc-id "$VPCs" --query GroupId --output text)
MYIP=$(curl -s https://checkip.amazonaws.com || echo 0.0.0.0)
aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG_ID" --protocol tcp --port 22 --cidr "${MYIP}/32" >/dev/null || true

AMI=$(aws ec2 describe-images --region "$REGION" --owners amazon \
  --filters "Name=name,Values=al2023-ami-2023*-x86_64" "Name=state,Values=available" \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
echo "AMI=$AMI"

USERDATA=$(cat <<EOF
#!/bin/bash
set -euxo pipefail
dnf install -y python3.11 python3.11-pip git tar
mkdir -p /opt/suraksha
aws s3 cp s3://$BUCKET/job.tgz /tmp/job.tgz --region $REGION
tar -C /opt/suraksha -xzf /tmp/job.tgz
cd /opt/suraksha
python3.11 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r aws_train/requirements_codemix.txt
# fix scorer path already relative
nohup bash aws_train/suraksha_codemix_loop.sh > /opt/suraksha/codemix_hinglish/ec2_supervisor.out 2>&1 &
echo \$! > /opt/suraksha/codemix_hinglish/ec2_supervisor.pid
EOF
)

IID=$(aws ec2 run-instances --region "$REGION" \
  --image-id "$AMI" --instance-type t3.large \
  --key-name "$KEY_NAME" --security-group-ids "$SG_ID" \
  --iam-instance-profile "Name=$PROFILE_NAME" \
  --user-data "$USERDATA" \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":40,"VolumeType":"gp3"}}]' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME}]" \
  --query 'Instances[0].InstanceId' --output text)

echo "Waiting for instance $IID running..."
aws ec2 wait instance-running --region "$REGION" --instance-ids "$IID"
PUB=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$IID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

cat > "$STATE" <<EOT
REGION=$REGION
ACCOUNT=$ACCOUNT
BUCKET=$BUCKET
INSTANCE_ID=$IID
PUBLIC_IP=$PUB
KEY_NAME=$KEY_NAME
KEY_FILE=/tmp/${KEY_NAME}.pem
ROLE_NAME=$ROLE_NAME
PROFILE_NAME=$PROFILE_NAME
SG_ID=$SG_ID
EOT

echo "EC2 launched."
echo "  instance: $IID"
echo "  ip: $PUB"
echo "  s3: s3://$BUCKET/codemix_hinglish/"
echo "  state: $STATE"
echo "Close laptop anytime. Pull results with:"
echo "  aws s3 sync s3://$BUCKET/codemix_hinglish ./codemix_hinglish --region $REGION"
