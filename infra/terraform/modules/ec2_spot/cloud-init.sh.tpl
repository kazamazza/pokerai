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
apt-get install -y git software-properties-common unzip curl jq \
  libgomp1 libstdc++6

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

# --- Install TexasSolver (Linux x86_64) and publish SOLVER_BIN ---
SOLVER_DIR="/opt/texas-solver"
VER="v0.2.0"   # change here when you want a newer release
ASSET="TexasSolver-$VER-Linux.zip"
URL="https://github.com/bupticybee/TexasSolver/releases/download/$VER/$ASSET"

if [ ! -x "$SOLVER_DIR/console_solver" ]; then
  log "Installing TexasSolver $VER …"
  mkdir -p "$SOLVER_DIR"
  curl -fsSL "$URL" -o /tmp/solver.zip
  unzip -oq /tmp/solver.zip -d "$SOLVER_DIR/"
  rm -f /tmp/solver.zip || true

  # find the console_solver inside the extracted folder
  BIN="$(find "$SOLVER_DIR" -maxdepth 2 -type f -name console_solver | head -n1 || true)"
  if [ -z "$BIN" ]; then
    log "[ERROR] console_solver not found after unzip"; ls -R "$SOLVER_DIR"; exit 1
  fi

  install -m 0755 "$BIN" "$SOLVER_DIR/console_solver"
  ln -sf "$SOLVER_DIR/console_solver" /usr/local/bin/texas-solver
fi

chmod 0755 "$SOLVER_DIR/console_solver"

# Make it visible to all users and future shells
grep -q '^SOLVER_BIN=' /etc/environment || echo "SOLVER_BIN=$SOLVER_DIR/console_solver" | sudo tee -a /etc/environment >/dev/null
# (optionally ensure /opt/texas-solver is on PATH for interactive shells)
if ! grep -q '/opt/texas-solver' /etc/environment; then
  echo 'PATH="/opt/texas-solver:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"' | sudo tee -a /etc/environment >/dev/null
fi
export SOLVER_BIN="$SOLVER_DIR/console_solver"

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

# Set global environment vars
echo "AWS_REGION=eu-central-1" | sudo tee -a /etc/environment
echo "AWS_DEFAULT_REGION=eu-central-1" | sudo tee -a /etc/environment
echo "AWS_BUCKET_NAME=pokeraistore" | sudo tee -a /etc/environment
echo "AWS_SQS_QUEUE_URL=$sqs_queue_url" | sudo tee -a /etc/environment
echo "AWS_SQS_DLQ_URL=$sqs_dlq_url" | sudo tee -a /etc/environment

# Optional: tag shell sessions
echo "export WORKER_TAG=${worker_name}" >> ~/.bashrc

# ===== Runtime setup & launch =====
set +u
set -o allexport
source /etc/environment || true
set +o allexport

# Decide process count: operator can export MAX_PROCS in /etc/environment.
# Escape ONLY this default expansion so Terraform doesn't interpolate it.
N="$${MAX_PROCS:-}"
if [ -z "$N" ]; then
  CORES="$(nproc || echo 1)"
  HEADROOM=1
  N=$(( CORES - HEADROOM ))
  [ "$N" -lt 1 ] && N=1
fi

# Don’t let native libs oversubscribe threads
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "[init] Launching $N worker processes..."

PIDS_FILE=/var/run/worker_pids.txt
mkdir -p /var/run
: > "$PIDS_FILE"

for i in $(seq 1 "$N"); do
  LOG="/var/log/worker_$i.log"
  nohup /home/ubuntu/pokerai/env/bin/python -u "$script_to_run" > "$LOG" 2>&1 &
  PID=$!
  echo "$PID" >> "$PIDS_FILE"
  echo "[init] started worker_$i pid $PID -> $LOG"
  sleep 0.2
done

STARTED="$(wc -l < "$PIDS_FILE" | tr -d ' ')"
echo "[init] Launched $STARTED workers."

# CloudWatch Log Upload (safe method without inline subshells)
REGION="eu-central-1"
IMDS_TOKEN=$(curl -fsS -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
INSTANCE_ID=$(curl -fsS -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" http://169.254.169.254/latest/meta-data/instance-id || echo unknown)

set +e
aws logs create-log-group --log-group-name "/spot-workers/${worker_name}" --region "$REGION" || true
aws logs create-log-stream --log-group-name "/spot-workers/${worker_name}" --log-stream-name "init-$INSTANCE_ID" --region "$REGION" || true
TIMESTAMP=$(date +%s%3N)
LOG_MSG=$(sed "s/\"/'/g" /var/log/cloud-init-output.log)
aws logs put-log-events \
  --log-group-name "/spot-workers/${worker_name}" \
  --log-stream-name "init-$INSTANCE_ID" \
  --region "$REGION" \
  --log-events timestamp=$TIMESTAMP,message="$LOG_MSG" || true
set -e

log "CloudWatch init logs pushed."
log "Initialization complete."
exit 0