#!/bin/bash
# scripts/reset_resources.sh

set -euo pipefail

source .env

# Validate required variables
: "${AWS_REGION:?AWS_REGION not set in .env}"
: "${AWS_SQS_QUEUE_URL:?AWS_SQS_QUEUE_URL not set in .env}"
: "${AWS_SQS_DLQ_URL:?AWS_SQS_DLQ_URL not set in .env}"
: "${AWS_BUCKET_NAME:?AWS_BUCKET_NAME not set in .env}"

log() {
  echo "[reset] $1"
}

log "Purging SQS queues..."
aws sqs purge-queue --region "$AWS_REGION" --queue-url "$AWS_SQS_QUEUE_URL" || log "Queue purge failed (may not exist yet)"
aws sqs purge-queue --region "$AWS_REGION" --queue-url "$AWS_SQS_DLQ_URL" || log "DLQ purge failed (may not exist yet)"

log "Emptying S3 bucket: $AWS_BUCKET_NAME"
aws s3 rm "s3://$AWS_BUCKET_NAME" --recursive || log "S3 bucket delete failed"

log "Reset complete."