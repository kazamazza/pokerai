#!/bin/bash
# cloud-init.sh.tpl

set -euo pipefail

# Variables passed from Terraform
github_token="${github_token}"
sqs_queue_url="${aws_sqs_queue_url}"
sqs_dlq_url="${aws_sqs_dlq_url}"
access_key_id="${aws_access_key_id}"
secret_access_key="${aws_secret_access_key}"
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


# Export config to global environment


# Set global environment vars
echo "AWS_REGION=eu-central-1" | sudo tee -a /etc/environment
echo "AWS_ACCESS_KEY_ID=$access_key_id" | sudo tee -a /etc/environment
echo "AWS_SECRET_ACCESS_KEY=$secret_access_key" | sudo tee -a /etc/environment
echo "AWS_BUCKET_NAME=pokeraistore" | sudo tee -a /etc/environment
echo "AWS_SQS_QUEUE_URL=$sqs_queue_url" | sudo tee -a /etc/environment
echo "AWS_SQS_DLQ_URL=$sqs_dlq_url" | sudo tee -a /etc/environment

# Optional: tag shell sessions
echo "export WORKER_TAG=${worker_name}" >> ~/.bashrc

# Load the env vars into this shell for current script to use
set -o allexport
source /etc/environment
set +o allexport

# Activate venv
source /home/ubuntu/pokerai/env/bin/activate

# How many processes? ~ physical cores (vCPUs/2). Fallback to at least 1.
VCPUS=$(nproc)                  # e.g. 8 on c5.xlarge (8 vCPU)
PHYS=$(( VCPUS / 2 ))
N=$(( PHYS > 0 ? PHYS : 1 ))

# Keep native libs from oversubscribing threads
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "[init] Launching $N worker processes (pinned to cores)..."

# Launch N processes, each pinned to its own core
core=0
for i in $(seq 1 "$N"); do
  # Pin to two vCPUs per process if you like (core and its HT sibling)
  taskset -c "$core" nohup python "$script_to_run" > "/var/log/worker_${i}.log" 2>&1 &
  core=$(( (core + 1) % VCPUS ))
done

log "Worker script '$script_to_run' launched in background."

# CloudWatch Log Upload (safe method without inline subshells)
REGION="eu-central-1"
INSTANCE_ID_CMD="curl -s http://169.254.169.254/latest/meta-data/instance-id"
echo "$INSTANCE_ID_CMD" > /tmp/meta_fetch.sh
chmod +x /tmp/meta_fetch.sh
INSTANCE_ID=$(/tmp/meta_fetch.sh)

aws logs create-log-group --log-group-name "/spot-workers/${worker_name}" --region "$REGION" || true
aws logs create-log-stream --log-group-name "/spot-workers/${worker_name}" --log-stream-name "init-$INSTANCE_ID" --region "$REGION" || true

TIMESTAMP=$(date +%s%3N)
LOG_MSG=$(sed "s/\"/'/g" /var/log/cloud-init-output.log)

aws logs put-log-events \
  --log-group-name "/spot-workers/${worker_name}" \
  --log-stream-name "init-$INSTANCE_ID" \
  --region "$REGION" \
  --log-events timestamp=$TIMESTAMP,message="$LOG_MSG" || true

log "CloudWatch init logs pushed."
log "Initialization complete."