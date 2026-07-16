#!/usr/bin/env bash
set -euo pipefail

repository_root="$(git rev-parse --show-toplevel)"
revision="$(git -C "$repository_root" rev-parse HEAD)"
result_dir="$repository_root/artifacts/continuum/docker"
mkdir -p "$result_dir"
status_path="$result_dir/status.json"
rm -f -- "$status_path"
if ! docker info >/dev/null 2>&1; then
  printf '%s\n' '{"schema_version":"sloforge.continuum.docker-status/v1","status":"unexercised","reason":"Docker daemon unavailable"}' > "$status_path"
  echo "Continuum Docker smoke unexercised: Docker daemon unavailable"
  exit 0
fi

image="sloforge-continuum-smoke:${revision:0:12}"
context_root="$(mktemp -d "${TMPDIR:-/tmp}/sloforge-continuum-docker.XXXXXX")"
cleanup() {
  exit_status=$?
  docker image rm --force "$image" >/dev/null 2>&1 || true
  rm -rf -- "$context_root"
  if test "$exit_status" -ne 0 && test ! -f "$status_path"; then
    printf '%s\n' '{"schema_version":"sloforge.continuum.docker-status/v1","status":"failed","reason":"Continuum Docker acceptance command failed"}' > "$status_path"
  fi
  return "$exit_status"
}
trap cleanup EXIT INT TERM
git -C "$repository_root" archive "$revision" | tar -x -C "$context_root"
docker build --file "$context_root/deploy/docker/Continuum.Dockerfile" --build-arg "SLOFORGE_SOURCE_COMMIT=$revision" --tag "$image" "$context_root"
docker run --rm --read-only --tmpfs /tmp:rw,noexec,nosuid,size=512m --security-opt no-new-privileges:true "$image" sloforge continuum migrate --mode pre-copy --seed 317 --output /tmp/continuum
printf '%s\n' "{\"schema_version\":\"sloforge.continuum.docker-status/v1\",\"status\":\"exercised\",\"revision\":\"$revision\"}" > "$status_path"
