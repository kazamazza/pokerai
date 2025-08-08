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

# Load /etc/environment
set -o allexport
source /etc/environment
set +o allexport

# Make sure current shell (and children) has the region
export AWS_DEFAULT_REGION=eu-central-1

# Start worker processes, one per vCPU
source /home/ubuntu/pokerai/env/bin/activate
command -v taskset >/dev/null 2>&1 || apt-get install -y --no-install-recommends util-linux
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

N=$(nproc)
for core in $(seq 0 $((N-1))); do
  WORKER_INDEX="$core" nohup /home/ubuntu/pokerai/env/bin/python -u "$script_to_run" \
    > "/var/log/worker_${core}.log" 2>&1 &
  pid=$!
  taskset -pc "$core" "$pid" >/dev/null 2>&1 || true
done

log "Launched $N workers for $N vCPUs."

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