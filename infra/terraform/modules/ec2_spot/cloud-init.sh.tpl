#!/bin/bash
# cloud-init.sh.tpl

# strict mode + line-level error tracing into logs
set -Eeuo pipefail
trap 'ec=$?; echo "[cloud-init:ERR] line $LINENO: $BASH_COMMAND (exit=$ec)" >&2; push_cw_init_log || true; exit $ec' ERR

# Variables passed from Terraform
github_token="${github_token}"
sqs_queue_url="${aws_sqs_queue_url}"
sqs_dlq_url="${aws_sqs_dlq_url}"
access_key_id="${aws_access_key_id}"
secret_access_key="${aws_secret_access_key}"
script_to_run="${script_to_run}"
worker_name="${worker_name}"

# unified logging
exec > >(tee /var/log/user-data.log | logger -t user-data -s 2>/dev/console) 2>&1

log(){ echo "[cloud-init] $*"; }

safe(){ "$@" || echo "[WARN] $* failed (ignored)"; }

retry(){
  local cmd=$1
  local max=$2
  if [ -z "$max" ]; then
    max=5
  fi
  local a=0
  while true; do
    eval "$cmd" && return 0
    a=$((a+1))
    if [ $a -ge $max ]; then
      return 1
    fi
    sleep $((2**a))
  done
}

# push cloud-init logs to CloudWatch (called on success and in trap)
push_cw_init_log(){
  REGION="$(printenv AWS_REGION 2>/dev/null)"
  if [ -z "$REGION" ]; then
    REGION="eu-central-1"
  fi

  GROUP="/spot-workers/${worker_name}"

  TOKEN="$(curl -fsS -X PUT "http://169.254.169.254/latest/api/token" \
           -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" || true)"
  IID="$(curl -fsS -H "X-aws-ec2-metadata-token: $TOKEN" \
          http://169.254.169.254/latest/meta-data/instance-id || echo unknown)"
  STREAM="init-$IID"

  aws logs create-log-group  --log-group-name "$GROUP" --region "$REGION" 2>/dev/null || true
  aws logs create-log-stream --log-group-name "$GROUP" --log-stream-name "$STREAM" --region "$REGION" 2>/dev/null || true

  TS="$(date +%s%3N)"
  MSG="$(sed 's/\"/'\''/g' /var/log/cloud-init-output.log 2>/dev/null || true)"

  if [ -n "$MSG" ]; then
    aws logs put-log-events \
      --log-group-name "$GROUP" \
      --log-stream-name "$STREAM" \
      --region "$REGION" \
      --log-events timestamp="$TS",message="$MSG" >/dev/null 2>&1 || true
  fi
}

log "Starting instance initialization."

retry "apt-get update -y" 4 || apt-get update -y
retry "apt-get -y upgrade" 2 || true
apt-get install -y git software-properties-common unzip curl jq

curl -fsSLo awscliv2.zip "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip"
unzip -q awscliv2.zip
./aws/install --update || ./aws/install || true

add-apt-repository ppa:deadsnakes/ppa -y
apt-get update -y
apt-get install -y python3.11 python3.11-venv python3.11-dev

PARTITION=$(findmnt -n -o SOURCE /)
ROOT_DEVICE=$(lsblk -no pkname "$PARTITION")
ROOT_DEVICE="/dev/$ROOT_DEVICE"
log "Root device: $ROOT_DEVICE"
log "Partition: $PARTITION"
log "Disk usage before resize:"; df -h /

safe growpart "$ROOT_DEVICE" 1
FS_TYPE=$(df -T / | tail -1 | awk '{print $2}')
if [[ "$FS_TYPE" == "ext4" ]]; then
  safe resize2fs "$PARTITION"
elif [[ "$FS_TYPE" == "xfs" ]]; then
  safe xfs_growfs /
else
  log "[WARN] Unknown FS type: $FS_TYPE"
fi
log "Disk usage after resize:"; df -h /

REPO_URL="https://x-access-token:$github_token@github.com/kazamazza/pokerai.git"
log "Cloning from: $(echo "$REPO_URL" | cut -c1-50)..."
cd /home/ubuntu || exit 1
if ! git clone "$REPO_URL"; then log "[ERROR] Git clone failed."; exit 1; fi
cd pokerai
python3.11 -m venv env || { log "venv failed"; exit 1; }
source env/bin/activate


pip install --upgrade pip || { log "pip upgrade failed"; exit 1; }
pip install -r requirements.txt || { log "requirements install failed"; exit 1; }

# Build env file for containers (instead of /etc/environment)
CONTAINER_ENV="/etc/pokerai.env"
sudo bash -c "cat > $CONTAINER_ENV" <<EOF
AWS_REGION=eu-central-1
AWS_DEFAULT_REGION=eu-central-1
AWS_BUCKET_NAME=pokeraistore
AWS_SQS_QUEUE_URL=$sqs_queue_url
AWS_SQS_DLQ_URL=$sqs_dlq_url
SOLVER_BIN=/opt/texas-solver/console_solver
WORKER_TAG=${worker_name}
MEM_LIMIT=6g
MAX_PROCS=
EOF
sudo chmod 0644 "$CONTAINER_ENV"

# Tag shell sessions with worker name
echo "export WORKER_TAG=${worker_name}" >> ~/.bashrc

# Docker
set +u
set -o allexport
source "$CONTAINER_ENV" || true
set +o allexport

log "Installing Docker..."
apt-get update -y
apt-get install -y ca-certificates curl gnupg
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | tee /etc/apt/keyrings/docker.asc >/dev/null
chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | tee /etc/apt/sources.list.d/docker.list >/dev/null
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
usermod -aG docker ubuntu || true
systemctl enable --now docker

# ECR login + pull with debug and retries (tag-aware)
ecr_repo="214061305689.dkr.ecr.eu-central-1.amazonaws.com/pokerai-worker"

# --- safe env helpers (no failing printenv) ---
getenv() { printenv "$1" 2>/dev/null || true; }
getenv_or_default() { v="$(getenv "$1")"; [ -n "$v" ] && printf '%s' "$v" || printf '%s' "$2"; }

# Determine tag (default to amd64 if IMAGE_TAG not provided)
IMAGE_TAG="$(getenv_or_default IMAGE_TAG amd64)"
ecr_image="$ecr_repo:$IMAGE_TAG"

debug_ecr(){
  local reg
  reg="$(echo "$ecr_repo" | cut -d/ -f1)"
  echo "[debug] identity:"; aws sts get-caller-identity || true
  echo "[debug] docker:"; systemctl is-active docker || true; docker info || sudo journalctl -u docker -n 120 --no-pager || true
  echo "[debug] ecr login -> $reg"
  aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$reg"
}

pull_with_retry(){ local img="$1"; retry "docker pull $img" 5; }

log "Logging into ECR..."
if ! debug_ecr; then echo "[cloud-init] [fatal] ECR login failed"; exit 1; fi

log "Pulling image: $ecr_image"
if ! pull_with_retry "$ecr_image"; then echo "[cloud-init] [fatal] docker pull failed for $ecr_image"; exit 1; fi

# ---- capacity planning (max out vCPU, cap by RAM) ----
TARGET_MEM_PER_WORKER="$(getenv_or_default TARGET_MEM_PER_WORKER 3g)"
RESERVE_MEM_GB="$(getenv_or_default RESERVE_MEM_GB 2)"

to_kb() {
  val="$1"
  lc="$(printf '%s' "$val" | tr '[:upper:]' '[:lower:]')"
  case "$lc" in
    *g) num="$(printf '%s' "$lc" | sed 's/[[:space:]]*g$//')"; echo $(( num * 1024 * 1024 ));;
    *m) num="$(printf '%s' "$lc" | sed 's/[[:space:]]*m$//')"; echo $(( num * 1024 ));;
    *k) num="$(printf '%s' "$lc" | sed 's/[[:space:]]*k$//')"; echo "$num";;
    *)   echo "$lc";;
  esac
}

# === SIMPLE CAPACITY: one worker per vCPU (MAX_PROCS can override) ===
CORES="$(nproc || echo 1)"
MAX_PROCS_VAL="$(getenv MAX_PROCS)"
if [ -n "$MAX_PROCS_VAL" ]; then
  N="$MAX_PROCS_VAL"
else
  N="$CORES"
fi
log "Launching $N workers (vCPU-based); MAX_PROCS can override."

# optional per-container memory cap (only if MEM_LIMIT is set)
MEM_LIMIT_VAL="$(getenv MEM_LIMIT)"
run_mem_flag=""
[ -n "$MEM_LIMIT_VAL" ] && run_mem_flag="-m $MEM_LIMIT_VAL --memory-swap -1"

# unbuffered Python logs
PYTHONUNBUFFERED="$(getenv_or_default PYTHONUNBUFFERED 1)"

for i in $(seq 1 "$N"); do
  NAME="worker_$i"
  LOG="/var/log/$${NAME}.log"   # terraform-safe

  # command
  cmd="python -u tools/rangenet/worker_flop.py --queue-url $sqs_queue_url --region $AWS_REGION"
  if [ -n "$sqs_dlq_url" ]; then
    cmd="$cmd --dlq-url $sqs_dlq_url"
  fi

  # envs for container
  docker_args="-e AWS_REGION=$AWS_REGION -e WORKER_TAG=$worker_name \
               -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
               -e MKL_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 \
               -e PYTHONUNBUFFERED=$PYTHONUNBUFFERED"
  [ -n "$access_key_id" ]     && docker_args="$docker_args -e AWS_ACCESS_KEY_ID=$access_key_id"
  [ -n "$secret_access_key" ] && docker_args="$docker_args -e AWS_SECRET_ACCESS_KEY=$secret_access_key"

  docker run -d --rm \
    --name "$NAME" \
    $run_mem_flag \
    --env-file /etc/pokerai.env \
    $docker_args \
    "$ecr_image" \
    $cmd

  echo "[init] started $NAME -> $LOG"
  ( docker logs -f "$NAME" > "$LOG" 2>&1 ) &
  sleep 0.2
done

log "All workers launched."
push_cw_init_log || true
log "Initialization complete."
exit 0