#!/bin/bash
# cloud-init.sh.tpl

set -euo pipefail

# GitHub token (passed from Terraform)
github_token="${github_token}"

exec > >(tee /var/log/user-data.log | logger -t user-data -s 2>/dev/console) 2>&1

log() {
  echo "[cloud-init] $1"
}

log "Starting instance initialization."

apt-get update -y && apt-get upgrade -y
apt-get install -y git software-properties-common awscli
add-apt-repository ppa:deadsnakes/ppa -y
apt-get update -y
apt-get install -y python3.11 python3.11-venv python3.11-dev

PARTITION=$(findmnt -n -o SOURCE /)
ROOT_DEVICE=$(lsblk -no pkname "$PARTITION")
ROOT_DEVICE="/dev/$ROOT_DEVICE"

log "Root device: $ROOT_DEVICE"
log "Partition: $PARTITION"

log "Disk usage before resize:"
df -h /

growpart "$ROOT_DEVICE" 1 || log "[WARN] growpart may already be applied"
resize2fs "$PARTITION" || log "[WARN] resize2fs may already be applied"

log "Disk usage after resize:"
df -h /


REPO_URL="https://x-access-token:$github_token@github.com/kazamazza/pokerai.git"
log "Cloning from: $(echo "$REPO_URL" | cut -c1-50)..."

cd /home/ubuntu || exit 1
if ! git clone "$REPO_URL"; then
  log "[ERROR] Git clone failed."
  exit 1
fi
cd pokerai

python3.11 -m venv env || { log "venv failed"; exit 1; }
source env/bin/activate
pip install --upgrade pip || { log "pip upgrade failed"; exit 1; }
pip install -r requirements.txt || { log "requirements install failed"; exit 1; }

nohup python3.11 workers/sqs_worker.py > worker.log 2>&1 &
log "Worker launched in background."

REGION="eu-central-1"
LOG_GROUP="/spot-workers/cloud-init"
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
LOG_STREAM="init-$INSTANCE_ID"

aws logs describe-log-groups --log-group-name-prefix "$LOG_GROUP" --region "$REGION" | grep "$LOG_GROUP" || \
  aws logs create-log-group --log-group-name "$LOG_GROUP" --region "$REGION"

aws logs describe-log-streams --log-group-name "$LOG_GROUP" --log-stream-name-prefix "$LOG_STREAM" --region "$REGION" | grep "$LOG_STREAM" || \
  aws logs create-log-stream --log-group-name "$LOG_GROUP" --log-stream-name "$LOG_STREAM" --region "$REGION"

TIMESTAMP=$(date +%s%3N)
LOG_CONTENT=$(cat /var/log/cloud-init-output.log | sed "s/\"/'/g")

aws logs put-log-events \
  --log-group-name "$LOG_GROUP" \
  --log-stream-name "$LOG_STREAM" \
  --region "$REGION" \
  --log-events timestamp=$TIMESTAMP,message="$LOG_CONTENT" || true

log "Initialization complete."
exit 0