#!/usr/bin/env bash
set -euo pipefail

repository_root="$(git rev-parse --show-toplevel)"
revision="$(git -C "$repository_root" rev-parse HEAD)"
image="sloforge-genesis-smoke:${revision:0:12}"

docker build \
  --file "$repository_root/deploy/docker/Genesis.Dockerfile" \
  --build-arg "SLOFORGE_SOURCE_COMMIT=$revision" \
  --tag "$image" \
  "$repository_root"
docker run --rm \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=256m \
  --security-opt no-new-privileges:true \
  "$image" \
  python -m sloforge.synthbench.demo \
    --output /tmp/synthbench \
    --seed 73129 \
    --count 2
