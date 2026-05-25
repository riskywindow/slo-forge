SHELL := /bin/bash
.DEFAULT_GOAL := check

.PHONY: bootstrap check demo benchmark-cpu benchmark-gpu docker-smoke package

bootstrap:
	@command -v uv >/dev/null 2>&1 || { echo "error: uv is required (https://docs.astral.sh/uv/)" >&2; exit 127; }
	@command -v cargo >/dev/null 2>&1 || { echo "error: cargo/rustup is required" >&2; exit 127; }
	@if test -f ui/package-lock.json; then command -v npm >/dev/null 2>&1 || { echo "error: Node.js/npm is required for ui/" >&2; exit 127; }; fi
	uv sync --locked --extra dev --extra deploy
	cargo fetch --locked
	cargo build --workspace --locked
	@if test -f ui/package-lock.json; then npm ci --prefix ui --cache "$(CURDIR)/.cache/npm"; fi

check:
	uv run --locked ruff format --check python tests
	uv run --locked ruff check python tests
	uv run --locked mypy python/sloforge
	uv run --locked pytest
	cargo fmt --all --check
	cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
	cargo test --workspace --all-features --locked
	@if test -f ui/package.json; then \
	  npm run typecheck --prefix ui && \
	  npm run lint --prefix ui && \
	  npm test --prefix ui && \
	  npm run build --prefix ui; \
	fi

demo:
	uv run --locked python -m sloforge.demo --artifact-dir artifacts/demo --report-dir reports/demo --reset

benchmark-cpu:
	uv run --locked python -m sloforge.demo --artifact-dir artifacts/cpu-demo --report-dir reports/cpu-demo --reset
	uv run --locked python -m sloforge.benchmarks finalize-cpu --source reports/cpu-demo --output reports

benchmark-gpu:
	uv run --locked python -m sloforge.benchmarks benchmark-gpu --output reports/gpu

package:
	uv build

docker-smoke:
	@set -euo pipefail; \
	  project=sloforge-smoke; \
	  cleanup() { docker compose -p "$$project" -f deploy/docker/compose.yaml down --volumes --remove-orphans >/dev/null 2>&1 || true; }; \
	  trap cleanup EXIT; \
	  docker compose -p "$$project" -f deploy/docker/compose.yaml up --build --wait --wait-timeout 180; \
	  curl --fail --silent --show-error http://127.0.0.1:18080/health >/dev/null; \
	  curl --fail --silent --show-error --no-buffer \
	    -H 'content-type: application/json' \
	    -d '{"model":"sloforge/mock","prompt":"docker smoke","max_tokens":3,"stream":true}' \
	    http://127.0.0.1:18080/v1/completions | grep -q '\[DONE\]'
