#!/usr/bin/env bash
# Deploy the static viewer to S3 (private) + CloudFront (HTTPS). Re-run `update` to push changes.
#   bash site/deploy_site.sh          first deploy (creates bucket + distribution)
#   bash site/deploy_site.sh update   re-upload files and invalidate cache
#   bash site/deploy_site.sh destroy  delete everything
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
STATE="$HERE/deploy_state.json"
REGION=us-east-1
cd "$HERE"

upload() {
  local B=$1
  python3 build_site.py >/dev/null
  local TMP; TMP=$(mktemp -d)
  mkdir -p "$TMP/data/cells"
  for f in data/manifest.json data/benchmark.json data/cells/*.json; do gzip -9 -c "$f" > "$TMP/$f"; done
  aws s3 sync "$TMP/data" "s3://$B/data" --content-encoding gzip --content-type application/json \
    --cache-control "public,max-age=3600" --delete --only-show-errors --region $REGION
  aws s3 cp index.html "s3://$B/index.html" --content-type "text/html; charset=utf-8" \
    --cache-control "no-cache" --only-show-errors --region $REGION
  printf 'User-agent: *\nDisallow: /\n' | aws s3 cp - "s3://$B/robots.txt" --content-type text/plain --only-show-errors --region $REGION
  rm -rf "$TMP"
}

if [ "${1:-}" = "destroy" ]; then
  B=$(python3 -c "import json;print(json.load(open('$STATE'))['bucket'])")
  D=$(python3 -c "import json;print(json.load(open('$STATE'))['distribution_id'])")
  ETAG=$(aws cloudfront get-distribution-config --id "$D" --query ETag --output text)
  aws cloudfront get-distribution-config --id "$D" --query DistributionConfig | python3 -c "import sys,json;d=json.load(sys.stdin);d['Enabled']=False;print(json.dumps(d))" > /tmp/dc.json
  aws cloudfront update-distribution --id "$D" --if-match "$ETAG" --distribution-config file:///tmp/dc.json >/dev/null
  echo "disabled $D; wait for Deployed then: aws cloudfront delete-distribution --id $D --if-match <etag>; aws s3 rb s3://$B --force"
  exit 0
fi

if [ "${1:-}" = "update" ]; then
  B=$(python3 -c "import json;print(json.load(open('$STATE'))['bucket'])")
  D=$(python3 -c "import json;print(json.load(open('$STATE'))['distribution_id'])")
  upload "$B"
  aws cloudfront create-invalidation --distribution-id "$D" --paths "/*" >/dev/null
  echo "updated"; exit 0
fi

ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
STAMP=$(date -u +%Y%m%d%H%M%S)
B="suraksha-viewer-${ACCOUNT}-${STAMP}"
TAGS='[{"Key":"Project","Value":"surakshanet-dataset-viewer"}]'

aws s3 mb "s3://$B" --region $REGION >/dev/null
aws s3api put-public-access-block --bucket "$B" --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws s3api put-bucket-tagging --bucket "$B" --tagging '{"TagSet":[{"Key":"Project","Value":"surakshanet-dataset-viewer"}]}'
upload "$B"

OAC=$(aws cloudfront create-origin-access-control --origin-access-control-config \
  "Name=oac-$B,SigningProtocol=sigv4,SigningBehavior=always,OriginAccessControlOriginType=s3" \
  --query OriginAccessControl.Id --output text)

cat > /tmp/cf.json <<JSON
{"DistributionConfig":{
 "CallerReference":"$STAMP","Comment":"SurakshaNet dataset viewer","Enabled":true,"DefaultRootObject":"index.html",
 "Origins":{"Quantity":1,"Items":[{"Id":"s3","DomainName":"$B.s3.$REGION.amazonaws.com","OriginAccessControlId":"$OAC",
   "S3OriginConfig":{"OriginAccessIdentity":""}}]},
 "DefaultCacheBehavior":{"TargetOriginId":"s3","ViewerProtocolPolicy":"redirect-to-https","Compress":true,
   "CachePolicyId":"658327ea-f89d-4fab-a63d-7e88639e58f6",
   "AllowedMethods":{"Quantity":2,"Items":["GET","HEAD"]}},
 "PriceClass":"PriceClass_100","HttpVersion":"http2and3"},
 "Tags":{"Items":[{"Key":"Project","Value":"surakshanet-dataset-viewer"}]}}
JSON
OUT=$(aws cloudfront create-distribution-with-tags --cli-input-json file:///tmp/cf.json)
D=$(echo "$OUT" | python3 -c "import sys,json;print(json.load(sys.stdin)['Distribution']['Id'])")
ARN=$(echo "$OUT" | python3 -c "import sys,json;print(json.load(sys.stdin)['Distribution']['ARN'])")
DOM=$(echo "$OUT" | python3 -c "import sys,json;print(json.load(sys.stdin)['Distribution']['DomainName'])")

aws s3api put-bucket-policy --bucket "$B" --policy "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Sid\":\"cf\",\"Effect\":\"Allow\",
 \"Principal\":{\"Service\":\"cloudfront.amazonaws.com\"},\"Action\":\"s3:GetObject\",\"Resource\":\"arn:aws:s3:::$B/*\",
 \"Condition\":{\"StringEquals\":{\"AWS:SourceArn\":\"$ARN\"}}}]}"

cat > "$STATE" <<JSON
{"bucket":"$B","distribution_id":"$D","domain":"$DOM","url":"https://$DOM","oac":"$OAC","tag":"Project=surakshanet-dataset-viewer"}
JSON
echo "URL: https://$DOM  (CloudFront takes ~5-10 min to finish deploying)"
