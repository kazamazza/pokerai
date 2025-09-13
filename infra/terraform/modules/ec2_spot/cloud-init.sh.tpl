#!/bin/bash
# cloud-init.sh.tpl

# Strict + trap to surface exact failure line and push logs
set -Eeuo pipefail
trap 'ec=$?; echo "[cloud-init:ERR] line $LINENO: $BASH_COMMAND (exit=$ec)" >&2; push_cw_init_log || true; exit $ec' ERR

# ===== Terraform-injected vars =====
github_token="${github_token}"
sqs_queue_url="${aws_sqs_queue_url}"
sqs_dlq_url="${aws_sqs_dlq_url}"
script_to_run="${script_to_run}"            # e.g. tools/rangenet/worker_flop.py
worker_name="${worker_name}"


exec > >(tee /var/log/user-data.log | logger -t user-data -s 2>/dev/console) 2>&1
log(){ echo "[cloud-init] $*"; }
safe(){ "$@" || echo "[WARN] $* failed (ignored)"; }


retry() {
  local cmd="$1"
  local max
  if [ $# -ge 2 ] && [ -n "$2" ]; then
    max="$2"
  else
    max=5
  fi
  local a=0
  while :; do
    eval "$cmd" && return 0
    a=$((a+1))
    [ "$a" -ge "$max" ] && return 1
    sleep $((2**a))
  done
}

# Push cloud-init logs to CloudWatch
push_cw_init_log(){
  local REGION; REGION="$(printenv AWS_REGION 2>/dev/null || echo eu-central-1)"
  local GROUP="/spot-workers/${worker_name}"
  local TOKEN; TOKEN="$(curl -fsS -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" || true)"
  local IID; IID="$(curl -fsS -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id || echo unknown)"
  local STREAM="init-$IID"

  aws logs create-log-group  --log-group-name "$GROUP" --region "$REGION" 2>/dev/null || true
  aws logs create-log-stream --log-group-name "$GROUP" --log-stream-name "$STREAM" --region "$REGION" 2>/dev/null || true

  local TS; TS="$(date +%s%3N)"
  local MSG; MSG="$(sed 's/\"/'\''/g' /var/log/user-data.log 2>/dev/null || true)"
  [ -n "$MSG" ] && aws logs put-log-events --log-group-name "$GROUP" --log-stream-name "$STREAM" --region "$REGION" --log-events timestamp="$TS",message="$MSG" >/dev/null 2>&1 || true
}

log "Starting instance initialization."

# ===== Base OS + AWS CLI + Python 3.11 =====
retry "apt-get update -y" 4 || apt-get update -y
retry "apt-get -y upgrade" 2 || true
apt-get install -y git software-properties-common unzip curl jq taskset util-linux
curl -fsSLo awscliv2.zip "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip"
unzip -q awscliv2.zip
./aws/install --update || ./aws/install || true

add-apt-repository -y ppa:deadsnakes/ppa
apt-get update -y
apt-get install -y python3.11 python3.11-venv python3.11-dev

# ===== Grow root FS (idempotent) =====
PARTITION=$(findmnt -n -o SOURCE /)
ROOT_DEVICE="/dev/$(lsblk -no pkname "$PARTITION")"
log "Root device: $ROOT_DEVICE  Partition: $PARTITION"
log "Disk usage before resize:"; df -h /
safe growpart "$ROOT_DEVICE" 1
FS_TYPE=$(df -T / | tail -1 | awk '{print $2}')
if [ "$FS_TYPE" = "ext4" ]; then safe resize2fs "$PARTITION"
elif [ "$FS_TYPE" = "xfs" ]; then safe xfs_growfs /
else log "[WARN] Unknown FS type: $FS_TYPE"
fi
log "Disk usage after resize:"; df -h /

# ===== Clone repo =====
REPO_URL="https://x-access-token:$github_token@github.com/kazamazza/pokerai.git"
cd /home/ubuntu
if [ ! -d pokerai ]; then
  log "Cloning repo…"
  if ! git clone "$REPO_URL" pokerai; then log "[ERROR] Git clone failed"; exit 1; fi
fi
cd pokerai

# ===== Python venv + deps =====
python3.11 -m venv env || { log "venv failed"; exit 1; }
# shellcheck disable=SC1091
. env/bin/activate
pip install --upgrade pip || { log "pip upgrade failed"; exit 1; }
pip install -r requirements.txt || { log "requirements install failed"; exit 1; }


SOLVER_DIR="/opt/texas-solver"
if [ ! -x "$SOLVER_DIR/console_solver" ]; then
  log "Installing TexasSolver..."
  mkdir -p "$SOLVER_DIR"

  TEXASSOLVER_VERSION="v0.2.0"
  ASSET="TexasSolver-$TEXASSOLVER_VERSION-Linux.zip"
  URL="https://github.com/bupticybee/TexasSolver/releases/download/$TEXASSOLVER_VERSION/$ASSET"

  curl -fSL "$URL" -o /tmp/solver.zip
  unzip -oq /tmp/solver.zip -d "$SOLVER_DIR/"
  rm -f /tmp/solver.zip || true

  # find placed binary
  BIN="$(find "$SOLVER_DIR" -maxdepth 2 -type f -name console_solver | head -n 1 || true)"
  if [ -z "$BIN" ]; then
    log "[ERROR] console_solver not found after unzip"
    ls -R "$SOLVER_DIR"
    exit 1
  fi

  install -m 0755 "$BIN" "$SOLVER_DIR/console_solver"
fi

chmod 0755 "$SOLVER_DIR/console_solver"
export SOLVER_BIN="$SOLVER_DIR/console_solver"

# ===== Export runtime env for this shell (and children) =====
export AWS_REGION="eu-central-1"
export AWS_DEFAULT_REGION="eu-central-1"
export AWS_SQS_QUEUE_URL="$sqs_queue_url"
export AWS_SQS_DLQ_URL="$sqs_dlq_url"
export WORKER_TAG="$worker_name"
export PYTHONUNBUFFERED=1

# ===== Worker scaling: one process per vCPU with headroom (defaults) =====
CORES="$(nproc || echo 1)"
HEADROOM="$(printenv HEADROOM 2>/dev/null || echo 1)"          # leave at least 1 core for OS/sshd
if [ "$HEADROOM" -lt 1 ]; then HEADROOM=1; fi
MAX_PROCS_ENV="$(printenv MAX_PROCS 2>/dev/null || true)"
if [ -n "$MAX_PROCS_ENV" ]; then
  N="$MAX_PROCS_ENV"
else
  N=$(( CORES - HEADROOM ))
  [ "$N" -lt 1 ] && N=1
fi

# Each worker should use exactly 1 thread to avoid oversubscription
THREADS_PER_WORKER="$(printenv THREADS_PER_WORKER 2>/dev/null || echo 1)"
log "CORES=$CORES  HEADROOM=$HEADROOM  N(workers)=$N  THREADS_PER_WORKER=$THREADS_PER_WORKER"

# Wait for sshd to be active before loading CPU
for _ in 1 2 3 4 5; do systemctl is-active ssh >/dev/null 2>&1 && break; sleep 2; done

# Gentle pre-launch guard: 1-min load < CORES
for _ in 1 2 3 4 5; do
  LA_INT="$(awk '{print int($1)}' /proc/loadavg)"
  [ "$LA_INT" -lt "$CORES" ] && break
  log "Host busy (load=$LA_INT); sleeping 5s…"; sleep 5
done

# ===== Launch N pinned workers (nice + taskset) =====
mkdir -p /var/log
PIDS_FILE="/var/run/pokerai_workers.pids"
: > "$PIDS_FILE"

for i in $(seq 1 "$N"); do
  # Pin away from CPU 0 if possible to keep OS/sshd responsive
  if [ "$CORES" -gt 1 ]; then
    CPU="$i"
    [ "$CPU" -ge "$CORES" ] && CPU=$(( (i % (CORES-1)) + 1 ))
  else
    CPU=0
  fi

  NAME="worker_$i"
  LOGF="/var/log/$NAME.log"

  # Exact command your worker expects
  CMD="python -u $script_to_run --queue-url \"$AWS_SQS_QUEUE_URL\" --region \"$AWS_REGION\" --threads \"$THREADS_PER_WORKER\""
  [ -n "$AWS_SQS_DLQ_URL" ] && CMD="$CMD --dlq-url \"$AWS_SQS_DLQ_URL\""

  # Run unbuffered, pinned, lower priority; tee to log and background
  bash -lc "exec nice -n 5 taskset -c $CPU stdbuf -oL -eL $CMD" >> "$LOGF" 2>&1 &

  PID=$!
  echo "$PID" >> "$PIDS_FILE"
  log "Started $NAME (pid=$PID, cpu=$CPU) -> $LOGF"
  sleep 0.5
done

# ===== Summary + keep running (no shutdown here) =====
log "Workers started: $(wc -l < "$PIDS_FILE") of $N expected."
ps -o pid,psr,ni,pcpu,pmem,cmd -p $(tr '\n' ' ' < "$PIDS_FILE") || true

push_cw_init_log || true
log "Initialization complete."
exit 0