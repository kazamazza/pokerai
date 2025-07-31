#!/bin/bash
# scripts/cloud-init.sh

set -e

# Update system and install dependencies
apt-get update -y && apt-get upgrade -y
apt-get install -y git python3-pip python3-venv

# Set working dir
cd /home/ubuntu

# GitHub token injected by Terraform (via user-data template)
GITHUB_TOKEN="${github_token}"
REPO_URL="https://${GITHUB_TOKEN}@github.com/kazamazza/pokerai.git"

# Clone repo
git clone $REPO_URL
cd pokerai

# Set up Python environment
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Run worker in background
nohup python3 workers/sqs_worker.py > worker.log 2>&1 &