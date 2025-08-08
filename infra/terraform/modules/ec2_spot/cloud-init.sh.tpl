#!/bin/bash
set -euo pipefail

# ===== Vars from Terraform =====
github_token="${github_token}"
sqs_queue_url="${aws_sqs_queue_url}"
sqs_dlq_url="${aws_sqs_dlq_url}"
access_key_id="${aws_access_key_id}"
secret_access_key="${aws_secret_access_key}"
script_to_run="${script_to_run}"
worker_name="${worker_name}"

exec > >(tee /var/log/user-data.log | logger -t user-data -s 2>/dev/console) 2>&1
log(){ echo "[cloud-init] $1"; }
log "Starting instance initialization."

# Base pkgs
export DEBIAN_FRONTEND=noninteractive
apt-get update -y && apt-get upgrade -y
apt-get install -y git software-properties-common unzip curl jq \
  cloud-guest-utils xfsprogs build-essential

# AWS CLI v2
curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip -q awscliv2.zip
./aws/install --update || ./aws/install

# Python 3.11
add-apt-repository ppa:deadsnakes/ppa -y
apt-get update -y
apt-get install -y python3.11 python3.11-venv python3.11-dev

# Resize root FS (best-effort)
PARTITION=$(findmnt -n -o SOURCE /)                      # e.g. /dev/nvme0n1p1
ROOT_DEVICE="/dev/$(lsblk -no pkname "$PARTITION")"      # e.g. /dev/nvme0n1
log "Root device: $ROOT_DEVICE  Partition: $PARTITION"
df -h /
growpart "$ROOT_DEVICE" 1 || log "[WARN] growpart failed/na"
FS_TYPE=$(df -T / | tail -1 | awk '{print $2}')
if [[ "$FS_TYPE" == "ext4" ]]; then
  resize2fs "$PARTITION" || log "[WARN] resize2fs skipped"
elif [[ "$FS_TYPE" == "xfs" ]]; then
  xfs_growfs / || log "[WARN] xfs_growfs skipped"
else
  log "[WARN] Unknown FS type: $FS_TYPE"
fi
df -h /

# Clone repo
REPO_URL="https://x-access-token:${github_token}@github.com/kazamazza/pokerai.git"
cd /home/ubuntu
git clone "$REPO_URL" || { log "[ERROR] Git clone failed"; exit 1; }
cd pokerai

# Venv + deps
python3.11 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# ===== Runtime env =====
# Write to /etc/environment so non-interactive shells inherit it
{
  echo "AWS_REGION=eu-central-1"
  echo "AWS_ACCESS_KEY_ID=${aws_access_key_id}"
  echo "AWS_SECRET_ACCESS_KEY=${aws_secret_access_key}"
  echo "AWS_BUCKET_NAME=pokeraistore"
  echo "AWS_SQS_QUEUE_URL=${aws_sqs_queue_url}"
  echo "AWS_SQS_DLQ_URL=${aws_sqs_dlq_url}"
  echo "WORKER_TAG=${worker_name}"               # <— restored
} | tee -a /etc/environment >/dev/null

# Load into current shell
set -o allexport
# shellcheck disable=SC1091
source /etc/environment || true
set +o allexport

# Ensure taskset exists
if ! command -v taskset >/dev/null 2>&1; then
  apt-get install -y --no-install-recommends util-linux
fi

# Compute process count (use all vCPUs)
N=$(nproc)

# Avoid thread oversubscription
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

log "[init] Launching $${N} worker processes (pinned)…"
cd /home/ubuntu/pokerai
source env/bin/activate

PIDFILE="/var/run/pokerai-workers.pids"
TOTAL_CPUS="$(nproc)"

if [ -f "$PIDFILE" ] && pgrep -F "$PIDFILE" >/dev/null 2>&1; then
  log "[init] Workers already running; skip relaunch."
else
  : > "$PIDFILE"
  core=0
  for i in $(seq 1 "$${N}"); do
    WORKER_INDEX="$${i}" taskset -c "$${core}" \
      nohup python "$script_to_run" > "/var/log/worker_$${i}.log" 2>&1 &
    echo $! >> "$PIDFILE"
    log "[init] worker_$${i} -> CPU $${core} (pid $!)"
    core=$(( (core + 1) % $TOTAL_CPUS ))
  done
fi

# CloudWatch upload (don’t crash if missing)
REGION="eu-central-1"
INSTANCE_ID="$(curl -fsS http://169.254.169.254/latest/meta-data/instance-id || echo unknown)"
aws logs create-log-group  --log-group-name "/spot-workers/${worker_name}" --region "$REGION" || true
aws logs create-log-stream --log-group-name "/spot-workers/${worker_name}" --log-stream-name "init-$INSTANCE_ID" --region "$REGION" || true
TIMESTAMP=$(date +%s%3N)
# Prefer the file we actually wrote to; guard with || true
LOG_MSG="$(sed 's/\"/'\''/g' /var/log/user-data.log || true)"
aws logs put-log-events \
  --log-group-name "/spot-workers/${worker_name}" \
  --log-stream-name "init-$INSTANCE_ID" \
  --region "$REGION" \
  --log-events timestamp=$TIMESTAMP,message="$LOG_MSG" || true

log "Initialization complete."