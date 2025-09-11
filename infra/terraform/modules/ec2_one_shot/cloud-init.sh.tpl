#!/bin/bash
# cloud-init.sh.tpl

set -euo pipefail

# Variables passed from Terraform
github_token="${github_token}"
sqs_queue_url="${aws_sqs_queue_url}"
script_to_run="${script_to_run}"
worker_name="${worker_name}"

exec > >(tee /var/log/user-data.log | logger -t user-data -s 2>/dev/console) 2>&1

log() {
  echo "[cloud-init] $1"
}

log "Starting instance initialization."

apt-get update -y && apt-get upgrade -y
apt-get install -y git software-properties-common unzip curl jq

curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -q awscliv2.zip
./aws/install --update

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

FS_TYPE=$(df -T / | tail -1 | awk '{print $2}')
if [[ "$FS_TYPE" == "ext4" ]]; then
  resize2fs "$PARTITION" || log "[WARN] resize2fs may already be applied"
elif [[ "$FS_TYPE" == "xfs" ]]; then
  xfs_growfs / || log "[WARN] xfs_growfs may already be applied"
else
  log "[WARN] Unknown FS type: $FS_TYPE"
fi

log "Disk usage after resize:"
df -h /

# --- earlier in cloud-init: install docker + awscli (you already do AWS CLI) ---
apt-get update -y
apt-get install -y docker.io
systemctl enable --now docker

AWS_REGION="eu-central-1"
AWS_ACCOUNT_ID="$(curl -s http://169.254.169.254/latest/dynamic/instance-identity/document | jq -r .accountId)"
REPO="pokerai"
IMAGE="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO"
TAG="worker-v1"   # set via Terraform var if you want
QUEUE_URL="${aws_sqs_queue_url}"
DLQ_URL="${aws_sqs_dlq_url}"    # if you have one
WORKER_TAG="${worker_name}"

# ECR login (instance role recommended)
aws ecr get-login-password --region "$AWS_REGION" \
| docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

# Pull the exact image/tag
docker pull "$IMAGE:$TAG"

# Decide process count
N="$${MAX_PROCS:-}"
if [ -z "$N" ]; then
  N="$(nproc || echo 1)"
fi

echo "[init] Launching $N docker workers from $IMAGE:$TAG ..."

for i in $(seq 1 "$N"); do
  LOG="/var/log/worker_$i.log"
  docker run -d --rm \
    --name "worker_$i" \
    -e AWS_REGION="$AWS_REGION" \
    -e AWS_SQS_QUEUE_URL="$QUEUE_URL" \
    -e AWS_SQS_DLQ_URL="$DLQ_URL" \
    -e WORKER_TAG="$WORKER_TAG-$i" \
    "$IMAGE:$TAG" \
    python tools/rangenet/worker_flop.py --queue-url "$QUEUE_URL" --region "$AWS_REGION" --threads 1 \
    > "$LOG" 2>&1 || true
  echo "[init] started worker_$i -> $LOG"
  sleep 0.2
done

# CloudWatch log push
REGION="eu-central-1"
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)

aws logs create-log-group --log-group-name "/spot-workers/${worker_name}" --region "$REGION" || true
aws logs create-log-stream --log-group-name "/spot-workers/${worker_name}" --log-stream-name "init-$INSTANCE_ID" --region "$REGION" || true

TIMESTAMP=$(date +%s%3N)
LOG_MSG=$(sed "s/\"/'/g" /var/log/user-data.log)

aws logs put-log-events \
  --log-group-name "/spot-workers/${worker_name}" \
  --log-stream-name "init-$INSTANCE_ID" \
  --region "$REGION" \
  --log-events timestamp=$TIMESTAMP,message="$LOG_MSG" || true

log "CloudWatch logs pushed. Preparing shutdown."

# Shutdown logic with retry
for i in {1..3}; do
  shutdown -h now && break
  log "Retrying shutdown ($i)..."
  sleep 10
done

log "Initialization complete."