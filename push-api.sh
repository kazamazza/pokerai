#!/usr/bin/env bash
set -euo pipefail

# --- config ---
ACCOUNT="214061305689"          # 👈 fill this in
REGION="eu-central-1"
REPO="pokerai-api"
TAG="latest"

# --- login ---
aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com

# --- build + push ---
docker buildx build \
  --platform linux/amd64 \
  --target api \
  -t $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO:$TAG \
  --push \
  .