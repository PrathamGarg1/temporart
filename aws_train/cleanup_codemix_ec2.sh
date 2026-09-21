#!/usr/bin/env bash
# Tear down EC2 worker created by launch_codemix_ec2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STATE="$ROOT/aws_train/codemix_ec2.env"
[[ -f "$STATE" ]] || { echo "missing $STATE"; exit 1; }
# shellcheck disable=SC1090
source "$STATE"
aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" >/dev/null || true
aws ec2 wait instance-terminated --region "$REGION" --instance-ids "$INSTANCE_ID" || true
aws ec2 delete-security-group --region "$REGION" --group-id "$SG_ID" 2>/dev/null || true
aws ec2 delete-key-pair --region "$REGION" --key-name "$KEY_NAME" 2>/dev/null || true
aws iam remove-role-from-instance-profile --instance-profile-name "$PROFILE_NAME" --role-name "$ROLE_NAME" 2>/dev/null || true
aws iam delete-instance-profile --instance-profile-name "$PROFILE_NAME" 2>/dev/null || true
aws iam delete-role-policy --role-name "$ROLE_NAME" --policy-name codemix-inline 2>/dev/null || true
aws iam delete-role --role-name "$ROLE_NAME" 2>/dev/null || true
echo "Terminated $INSTANCE_ID. S3 bucket kept: s3://$BUCKET (delete manually if desired)"
