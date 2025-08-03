#!/bin/bash
# scripts/reset_resources.sh

set -euo pipefail

source .env

: "${AWS_REGION:?AWS_REGION not set in .env}"
: "${AWS_BUCKET_NAME:?AWS_BUCKET_NAME not set in .env}"

# These can be in .env or hardcoded if static
: "${PRE_FLOP_QUEUE_URL:?}"
: "${EQUITY_QUEUE_URL:?}"
: "${EXPLOIT_QUEUE_URL:?}"

# Optional DLQs (silently skip if not set)
PRE_FLOP_DLQ_URL=${PRE_FLOP_DLQ_URL:-}
EQUITY_DLQ_URL=${EQUITY_DLQ_URL:-}
EXPLOIT_DLQ_URL=${EXPLOIT_DLQ_URL:-}

log() {
  echo "[reset] $1"
}

purge_queue() {
  local url=$1
  local name=$2
  if [[ -n "$url" ]]; then
    log "Purging $name ..."
    aws sqs purge-queue --region "$AWS_REGION" --queue-url "$url" || log "$name purge failed"
  fi
}

log "Purging primary queues..."
purge_queue "$PRE_FLOP_QUEUE_URL" "Preflop Queue"
purge_queue "$EQUITY_QUEUE_URL" "Equity Queue"
purge_queue "$EXPLOIT_QUEUE_URL" "Exploit Queue"

log "Purging DLQs..."
purge_queue "$PRE_FLOP_DLQ_URL" "Preflop DLQ"
purge_queue "$EQUITY_DLQ_URL" "Equity DLQ"
purge_queue "$EXPLOIT_DLQ_URL" "Exploit DLQ"

log "Emptying S3 bucket: $AWS_BUCKET_NAME"
aws s3 rm "s3://$AWS_BUCKET_NAME" --recursive || log "S3 bucket delete failed"

log "✅ Reset complete."