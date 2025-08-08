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

# ===== Runtime setup & launch =====

source /home/ubuntu/pokerai/env/bin/activate

#    These come from Terraform templatefile() vars.
echo "AWS_REGION=eu-central-1"                   | sudo tee -a /etc/environment >/dev/null
echo "AWS_ACCESS_KEY_ID=${aws_access_key_id}"    | sudo tee -a /etc/environment >/dev/null
echo "AWS_SECRET_ACCESS_KEY=${aws_secret_access_key}" | sudo tee -a /etc/environment >/dev/null
echo "AWS_BUCKET_NAME=pokeraistore"              | sudo tee -a /etc/environment >/dev/null
echo "AWS_SQS_QUEUE_URL=${aws_sqs_queue_url}"    | sudo tee -a /etc/environment >/dev/null
echo "AWS_SQS_DLQ_URL=${aws_sqs_dlq_url}"        | sudo tee -a /etc/environment >/dev/null
echo "WORKER_TAG=${worker_name}"                 | sudo tee -a /etc/environment >/dev/null

# Load /etc/environment into current shell
set -o allexport
# shellcheck disable=SC1091
source /etc/environment || true
set +o allexport

# --- Launch pinned worker processes ---

# Ensure taskset exists (non-interactive)
export DEBIAN_FRONTEND=noninteractive
if ! command -v taskset >/dev/null 2>&1; then
  apt-get update -y && apt-get install -y --no-install-recommends util-linux
fi

# Be sure we're in the repo root (for relative script paths)
cd /home/ubuntu/pokerai || exit 1
# Activate venv
# shellcheck disable=SC1091
source /home/ubuntu/pokerai/env/bin/activate

# Operator override via /etc/environment (MAX_PROCS), else ~physical cores
if [ -n "$${MAX_PROCS:-}" ]; then
  N="$${MAX_PROCS}"
else
  VCPUS="$(nproc)"
  PHYS=$(( VCPUS / 2 ))
  N=$(( PHYS > 0 ? PHYS : 1 ))
fi

# Prevent native libs from oversubscribing threads
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "[init] Launching $${N} worker processes (pinned to distinct CPUs)…"

PIDFILE="/var/run/pokerai-workers.pids"
TOTAL_CPUS="$(nproc)"

# Skip if already running (idempotent)
if [ -f "$PIDFILE" ] && pgrep -F "$PIDFILE" >/dev/null 2>&1; then
  echo "[init] Workers already running; skipping relaunch."
else
  : > "$PIDFILE"
  core=0
  for i in $(seq 1 "$${N}"); do
    WORKER_INDEX="$${i}" taskset -c "$${core}" \
      nohup python "${script_to_run}" > "/var/log/worker_$${i}.log" 2>&1 &
    echo $! >> "$PIDFILE"
    echo "[init] worker_$${i} -> CPU $${core} (pid $!)"
    core=$(( (core + 1) % TOTAL_CPUS ))
  done
fi

echo "[init] Workers launched. tail -f /var/log/worker_*.log"

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