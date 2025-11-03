#!/usr/bin/env bash
set -euo pipefail

DLQ_URL="https://sqs.eu-central-1.amazonaws.com/241514805921/Postflop-DLQ"
MAIN_URL="https://sqs.eu-central-1.amazonaws.com/241514805921/Postflop"

echo "Starting DLQ requeue..."

while true; do
  MSGS=$(aws sqs receive-message --queue-url "$DLQ_URL" \
    --max-number-of-messages 10 --visibility-timeout 0 \
    --wait-time-seconds 0 --output json)
  COUNT=$(echo "$MSGS" | jq '.Messages | length')
  if [ "$COUNT" == "null" ] || [ "$COUNT" -eq 0 ]; then
    echo "✅ Done — DLQ empty."
    break
  fi

  for i in $(seq 0 $((COUNT-1))); do
    BODY=$(echo "$MSGS" | jq -r ".Messages[$i].Body")
    RECEIPT=$(echo "$MSGS" | jq -r ".Messages[$i].ReceiptHandle")
    aws sqs send-message --queue-url "$MAIN_URL" --message-body "$BODY"
    aws sqs delete-message --queue-url "$DLQ_URL" --receipt-handle "$RECEIPT"
    echo "Requeued message $((i+1))/$COUNT"
  done
done