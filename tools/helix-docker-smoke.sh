#!/usr/bin/env bash
set -euo pipefail

repository_root="$(git rev-parse --show-toplevel)"
revision="$(git -C "$repository_root" rev-parse HEAD)"
result_dir="$repository_root/artifacts/helix/docker"
mkdir -p "$result_dir"
status_path="$result_dir/status.json"
if ! docker info >/dev/null 2>&1; then
  printf '%s\n' '{"schema_version":"sloforge.helix.docker-status/v1","status":"unexercised","reason":"Docker daemon unavailable"}' > "$status_path"
  echo "Helix Docker smoke unexercised: Docker daemon unavailable"
  exit 0
fi

image="sloforge-helix-smoke:${revision:0:12}"
context_root="$(mktemp -d "${TMPDIR:-/tmp}/sloforge-helix-docker.XXXXXX")"
cleanup() {
  exit_status=$?
  docker image rm --force "$image" >/dev/null 2>&1 || true
  rm -rf -- "$context_root"
  if test "$exit_status" -ne 0; then
    printf '%s\n' '{"schema_version":"sloforge.helix.docker-status/v1","status":"failed","reason":"Helix Docker acceptance command failed"}' > "$status_path"
  fi
  return "$exit_status"
}
trap cleanup EXIT INT TERM
git -C "$repository_root" archive "$revision" | tar -x -C "$context_root"
docker build --file "$context_root/deploy/docker/Helix.Dockerfile" \
  --build-arg "SLOFORGE_SOURCE_COMMIT=$revision" --tag "$image" "$context_root"
docker run --rm --read-only --tmpfs /tmp:rw,nosuid,size=1g \
  "$image" sloforge helix demo --seed 41 --output /tmp/helix-demo
printf '%s\n' "{\"schema_version\":\"sloforge.helix.docker-status/v1\",\"status\":\"exercised\",\"revision\":\"$revision\"}" > "$status_path"
