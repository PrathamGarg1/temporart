#!/usr/bin/env bash
# Publish the dataset viewer as a Hugging Face Docker Space.
#
#   hf auth login                                   # once, token with write access
#   bash viewer/deploy_space.sh public|private [space-name]
#
# Free CPU Space; no AWS cost. Data is copied into the Space repo (about 124 MB).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
VIS="${1:-private}"
NAME="${2:-surakshanet-threads}"
USER_NAME=$(hf auth whoami 2>/dev/null | head -1 | awk '{print $NF}')
[ -n "$USER_NAME" ] && [ "$USER_NAME" != "in" ] || { echo "run: hf auth login"; exit 1; }
REPO="$USER_NAME/$NAME"

STAGE=$(mktemp -d)
mkdir -p "$STAGE/data"
cp "$HERE/viewer/app.py" "$STAGE/app.py"
cp "$HERE"/dataset/all/{train,val,test}.jsonl "$STAGE/data/"
cat > "$STAGE/Dockerfile" <<'DOCKER'
FROM python:3.12-slim
WORKDIR /app
COPY app.py .
COPY data ./data
ENV DATA_DIR=/app/data PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
DOCKER
cat > "$STAGE/README.md" <<META
---
title: $NAME
sdk: docker
app_port: 7860
pinned: false
---
META

FLAG=""; [ "$VIS" = "private" ] && FLAG="--private"
hf repos create "$REPO" --repo-type space --space-sdk docker $FLAG --exist-ok
hf upload "$REPO" "$STAGE" . --repo-type space --commit-message "dataset viewer"
rm -rf "$STAGE"
echo "https://huggingface.co/spaces/$REPO   (build takes 2-4 minutes)"
