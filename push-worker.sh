#!/usr/bin/env bash
set -euo pipefail

# ---- config (env-overridable) ----
: "${ACCOUNT:=214061305689}"             # your AWS account ID
: "${REGION:=eu-central-1}"              # AWS region
: "${REPO:=pokerai-worker}"              # ECR repo name
: "${TAG:=latest}"                       # main tag
: "${PLATFORMS:=linux/amd64}"            # buildx platforms (multi-arch: "linux/amd64,linux/arm64")

# ---- derived ----
ECR_HOST="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE="${ECR_HOST}/${REPO}"
GIT_SHA="$(git rev-parse --short=12 HEAD 2>/dev/null || echo no-git)"
DATE_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ---- sanity checks ----
command -v aws >/dev/null || { echo "aws CLI not found"; exit 1; }
command -v docker >/dev/null || { echo "docker not found"; exit 1; }
docker buildx version >/dev/null 2>&1 || { echo "docker buildx not available"; exit 1; }

# ---- ensure ECR repo exists ----
if ! aws ecr describe-repositories --region "$REGION" --repository-names "$REPO" >/dev/null 2>&1; then
  aws ecr create-repository --region "$REGION" --repository-name "$REPO" >/dev/null
fi

# ---- login ----
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$ECR_HOST"

# ---- labels (OCI) ----
LABELS=(
  "--label" "org.opencontainers.image.title=${REPO}"
  "--label" "org.opencontainers.image.source=$(git config --get remote.origin.url 2>/dev/null || echo unknown)"
  "--label" "org.opencontainers.image.revision=${GIT_SHA}"
  "--label" "org.opencontainers.image.created=${DATE_UTC}"
)

# ---- build & push (dual-tag: explicit TAG and git SHA) ----
docker buildx build \
  --platform "${PLATFORMS}" \
  "${LABELS[@]}" \
  -t "${IMAGE}:${TAG}" \
  -t "${IMAGE}:${GIT_SHA}" \
  --target worker \
  --push \
  .

echo "✅ Pushed:"
echo "  ${IMAGE}:${TAG}"
echo "  ${IMAGE}:${GIT_SHA}"