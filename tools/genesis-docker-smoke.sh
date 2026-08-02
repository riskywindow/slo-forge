#!/usr/bin/env bash
set -euo pipefail

repository_root="$(git rev-parse --show-toplevel)"
revision="$(git -C "$repository_root" rev-parse HEAD)"
image="sloforge-genesis-smoke:${revision:0:12}"
container="sloforge-genesis-smoke-${revision:0:12}"
cleanup() {
  docker container rm --force "$container" >/dev/null 2>&1 || true
  docker image rm --force "$image" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

if ! docker info >/dev/null 2>&1; then
  echo "error: Docker daemon is unavailable; Genesis Docker smoke was not exercised" >&2
  exit 1
fi

docker build \
  --file "$repository_root/deploy/docker/Genesis.Dockerfile" \
  --build-arg "SLOFORGE_SOURCE_COMMIT=$revision" \
  --tag "$image" \
  "$repository_root"
docker run --rm \
  --name "$container" \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=256m \
  --security-opt no-new-privileges:true \
  "$image" \
  python -m sloforge.synthbench.demo \
    --output /tmp/synthbench \
    --seed 73129 \
    --count 2
