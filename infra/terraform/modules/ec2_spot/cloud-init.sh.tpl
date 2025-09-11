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
echo "AWS_DEFAULT_REGION=eu-central-1" | sudo tee -a /etc/environment
echo "AWS_BUCKET_NAME=pokeraistore" | sudo tee -a /etc/environment
echo "AWS_SQS_QUEUE_URL=$sqs_queue_url" | sudo tee -a /etc/environment
echo "AWS_SQS_DLQ_URL=$sqs_dlq_url" | sudo tee -a /etc/environment

# Optional: tag shell sessions
echo "export WORKER_TAG=${worker_name}" >> ~/.bashrc

# ===== Runtime setup & launch (Docker) =====
set +u
set -o allexport
source /etc/environment || true
set +o allexport

log "Installing Docker..."
apt-get update -y
apt-get install -y ca-certificates curl gnupg
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | tee /etc/apt/keyrings/docker.asc >/dev/null
chmod a+r /etc/apt/keyrings/docker.asc
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  | tee /etc/apt/sources.list.d/docker.list >/dev/null
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
usermod -aG docker ubuntu || true
systemctl enable --now docker

# ECR login
ecr_image="214061305689.dkr.ecr.eu-central-1.amazonaws.com/pokerai-worker"
log "Logging into ECR..."
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$(echo "$ecr_image" | cut -d/ -f1)"

log "Pulling image: $ecr_image"
docker pull "$ecr_image"

# Process count (MAX_PROCS overrides nproc)
# Determine process count without bash parameter expansion to keep Terraform happy
if env | grep -q '^MAX_PROCS='; then
  N="$(printenv MAX_PROCS)"
else
  N="$(nproc || echo 1)"
fi
log "Launching $N worker containers..."

# Start N containers
for i in $(seq 1 "$N"); do
  NAME="worker_$i"
  LOG="/var/log/$$${NAME}.log"

  # Build the command
  CMD=(python tools/rangenet/worker_flop.py
       --queue-url "$sqs_queue_url"
       --region "$AWS_REGION")

  # Add DLQ only if provided
  if [ -n "$sqs_dlq_url" ]; then
    CMD+=(--dlq-url "$sqs_dlq_url")
  fi

  NAME="worker_$i"
LOG="/var/log/$${NAME}.log"   # escape for Terraform; expand in Bash

DOCKER_ENV_ARGS=()
if [ -n "$access_key_id" ]; then
  DOCKER_ENV_ARGS+=(-e "AWS_ACCESS_KEY_ID=$access_key_id")
fi
if [ -n "$secret_access_key" ]; then
  DOCKER_ENV_ARGS+=(-e "AWS_SECRET_ACCESS_KEY=$secret_access_key")
fi

if [ -n "$CPU_LIMIT" ]; then
  CPU_LIMIT="$CPU_LIMIT"
else
  CPU_LIMIT="2"
fi
if [ -n "$MEM_LIMIT" ]; then
  MEM_LIMIT="$MEM_LIMIT"
else
  MEM_LIMIT="3g"
fi

docker run -d --rm \
  --name "$NAME" \
  -e AWS_REGION="$AWS_REGION" \
  -e WORKER_TAG="$worker_name" \
  -e OMP_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  -e NUMEXPR_NUM_THREADS=1 \
  --cpus="$CPU_LIMIT" \
  -m "$MEM_LIMIT" \
  "${DOCKER_ENV_ARGS[@]}" \
  "$ecr_image" \
  "$${CMD[@]}"

echo "[init] started $NAME -> $LOG"
( docker logs -f "$NAME" > "$LOG" 2>&1 ) &
sleep 0.2
done

log "All workers launched."

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
