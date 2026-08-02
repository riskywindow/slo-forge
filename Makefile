SHELL := /bin/bash
.DEFAULT_GOAL := check

.PHONY: bootstrap check demo benchmark-cpu benchmark-gpu docker-smoke package \
	fabric-check fabric-demo autopsy-demo forgeci-demo warmpath-demo \
	extension-evaluation clean-room-test genesis-check genesis-demo \
	genesis-zero-day-demo genesis-redteam-demo genesis-evolution-demo \
	synthbench-smoke synthbench-evaluation genesis-evaluation \
	genesis-docker-smoke genesis-clean-room-test

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

fabric-check:
	uv run --locked ruff format --check python tests
	uv run --locked ruff check python tests
	uv run --locked mypy python/sloforge
	uv run --locked pytest -q tests/python/test_fabric*.py tests/python/test_autopsy*.py tests/python/test_forgeci*.py tests/python/test_warmpath*.py
	cargo fmt --all --check
	cargo clippy -p sloforge-fabric-protocol -p sloforge-fabric-sim --all-targets --all-features --locked -- -D warnings
	cargo test -p sloforge-fabric-protocol -p sloforge-fabric-sim --all-features --locked

fabric-demo:
	uv run --locked python -m sloforge.fabric.demo --artifact-dir artifacts/fabric-demo --report-dir reports/fabric-demo --reset

autopsy-demo:
	uv run --locked python -m sloforge.fabric.demo --artifact-dir artifacts/autopsy-demo --report-dir reports/autopsy-demo --reset

forgeci-demo:
	uv run --locked python -m sloforge.forgeci.demo --output artifacts/forgeci/demo --report reports/forgeci-evaluation.md --reset

warmpath-demo:
	uv run --locked python -m sloforge.warmpath.demo --artifact-dir artifacts/warmpath --report reports/warmpath-demo.md --reset

extension-evaluation:
	uv run --locked python -m sloforge.fabric.evaluation --artifact-dir artifacts/fabric/evaluation --report-dir reports --reset
	uv run --locked python -m sloforge.warmpath.evaluation --output artifacts/warmpath/evaluation --report reports/warmpath-evaluation.md --reset
	uv run --locked python -m sloforge.forgeci.demo --output artifacts/forgeci/demo --report reports/forgeci-evaluation.md --reset

genesis-check:
	uv run --locked ruff format --check python tests
	uv run --locked ruff check python tests
	uv run --locked mypy python/sloforge
	uv run --locked pytest -q tests/python/test_genesis*.py tests/python/test_synthbench.py
	cargo fmt --all --check
	cargo clippy -p sloforge-genesis-ir -p sloforge-genesis-modelcheck --all-targets --all-features --locked -- -D warnings
	cargo test -p sloforge-genesis-ir -p sloforge-genesis-modelcheck --all-features --locked

genesis-demo:
	uv run --locked python -m sloforge.genesis.demo --output artifacts/genesis/demo --seed 73129 --reset

genesis-zero-day-demo:
	uv run --locked python -m sloforge.genesis.demo --output artifacts/genesis/zero-day-demo --seed 73129 --reset

genesis-redteam-demo:
	uv run --locked python -m sloforge.redteam.demo --output artifacts/genesis/redteam-demo --seed 73129

genesis-evolution-demo:
	uv run --locked python -m sloforge.genesis.demo --output artifacts/genesis/evolution-demo --seed 73131 --reset

synthbench-smoke:
	uv run --locked python -m sloforge.synthbench.demo --output artifacts/synthbench/smoke --seed 73129 --count 2 --reset

synthbench-evaluation:
	uv run --locked python -m sloforge.synthbench.demo --output artifacts/synthbench/evaluation --seed 73129 --count 10 --reset

genesis-evaluation:
	uv run --locked python -m sloforge.genesis.evaluation --output artifacts/genesis/evaluation --seed 73129 --count 3 --reset

clean-room-test:
	./tools/clean-room-fabric.sh

genesis-clean-room-test:
	./tools/clean-room-genesis.sh

genesis-docker-smoke:
	./tools/genesis-docker-smoke.sh

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
