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

# Export config for this shell
export AWS_REGION=eu-central-1
export AWS_SQS_QUEUE_URL="$sqs_queue_url"
export WORKER_TAG="$worker_name"

# Run script and wait
log "Running script: $script_to_run"
python3.11 "$script_to_run"

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