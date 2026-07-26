SHELL := /bin/bash
.DEFAULT_GOAL := check
CARGO_BOUNDED_ENV := CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0

.PHONY: bootstrap check demo benchmark-cpu benchmark-gpu docker-smoke package \
	fabric-check fabric-demo autopsy-demo forgeci-demo warmpath-demo \
	extension-evaluation clean-room-test genesis-check genesis-demo \
	genesis-zero-day-demo genesis-redteam-demo genesis-evolution-demo \
	synthbench-smoke synthbench-evaluation genesis-evaluation \
	genesis-docker-smoke genesis-clean-room-test continuum-check continuum-demo \
	continuum-migration-demo continuum-fault-demo continuum-fork-demo \
	continuum-compatibility-demo continuum-benchmark-cpu continuum-benchmark-gpu \
	continuum-docker-smoke continuum-clean-room-test helix-check helix-demo \
	helix-branch-demo helix-replay-demo helix-training-demo helix-resource-demo \
	helix-promotion-demo helix-fault-demo helix-evaluation helix-docker-smoke \
	helix-clean-room-test branchfabric-trace-check \
	branchfabric-characterization-cpu branchfabric-characterization-gpu \
	branchfabric-characterization branchfabric-requirements branchfabric-report \
	branchfabric-clean-room-test

bootstrap:
	@command -v uv >/dev/null 2>&1 || { echo "error: uv is required (https://docs.astral.sh/uv/)" >&2; exit 127; }
	@command -v cargo >/dev/null 2>&1 || { echo "error: cargo/rustup is required" >&2; exit 127; }
	@if test -f ui/package-lock.json; then command -v npm >/dev/null 2>&1 || { echo "error: Node.js/npm is required for ui/" >&2; exit 127; }; fi
	uv sync --locked --extra dev --extra deploy
	cargo fetch --locked
	$(CARGO_BOUNDED_ENV) cargo build --workspace --locked
	@if test -f ui/package-lock.json; then npm ci --prefix ui --cache "$(CURDIR)/.cache/npm"; fi

check:
	uv run --locked ruff format --check python tests
	uv run --locked ruff check python tests
	uv run --locked mypy python/sloforge
	uv run --locked pytest
	cargo fmt --all --check
	$(CARGO_BOUNDED_ENV) cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
	$(CARGO_BOUNDED_ENV) cargo test --workspace --all-features --locked
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
	$(CARGO_BOUNDED_ENV) cargo clippy -p sloforge-fabric-protocol -p sloforge-fabric-sim --all-targets --all-features --locked -- -D warnings
	$(CARGO_BOUNDED_ENV) cargo test -p sloforge-fabric-protocol -p sloforge-fabric-sim --all-features --locked

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
	$(CARGO_BOUNDED_ENV) cargo clippy -p sloforge-genesis-ir -p sloforge-genesis-modelcheck --all-targets --all-features --locked -- -D warnings
	$(CARGO_BOUNDED_ENV) cargo test -p sloforge-genesis-ir -p sloforge-genesis-modelcheck --all-features --locked

genesis-demo:
	uv run --locked python -m sloforge.genesis.demo --output artifacts/genesis/demo --seed 73129 --reset

genesis-zero-day-demo:
	uv run --locked python -m sloforge.genesis.demo --output artifacts/genesis/zero-day-demo --seed 73129 --reset

genesis-redteam-demo:
	uv run --locked python -m sloforge.redteam.demo --output artifacts/genesis/redteam-demo --seed 73129 --reset

genesis-evolution-demo:
	uv run --locked python -m sloforge.genesis.demo --output artifacts/genesis/evolution-demo --seed 73131 --runtime-seed 73129 --reset

synthbench-smoke:
	uv run --locked python -m sloforge.synthbench.demo --output artifacts/synthbench/smoke --seed 73129 --count 2 --reset

synthbench-evaluation:
	uv run --locked python -m sloforge.synthbench.demo --output artifacts/synthbench/evaluation --seed 73129 --count 10 --reset

genesis-evaluation:
	uv run --locked python -m sloforge.genesis.evaluation_suite run --output artifacts/genesis/evaluation --seed 73129 --core-runs 3 --campaign-seeds 5 --h1-tasks 5 --reset

clean-room-test:
	./tools/clean-room-fabric.sh

genesis-clean-room-test:
	./tools/clean-room-genesis.sh

genesis-docker-smoke:
	./tools/genesis-docker-smoke.sh

continuum-check:
	uv run --locked ruff format --check python/sloforge/continuum python/sloforge/cli/continuum.py tests/python/test_continuum*.py
	uv run --locked ruff check python/sloforge/continuum python/sloforge/cli/continuum.py tests/python/test_continuum*.py
	uv run --locked mypy python/sloforge/continuum python/sloforge/cli/continuum.py
	uv run --locked pytest -q tests/python/test_continuum*.py
	cargo fmt --all --check
	$(CARGO_BOUNDED_ENV) cargo clippy -p sloforge-continuum-ir -p sloforge-state-transaction -p sloforge-state-modelcheck --all-targets --all-features --locked -- -D warnings
	$(CARGO_BOUNDED_ENV) cargo test -p sloforge-continuum-ir -p sloforge-state-transaction -p sloforge-state-modelcheck --all-features --locked

continuum-demo:
	uv run --locked sloforge continuum migrate --mode pre-copy --seed 317 --output artifacts/continuum/demo --reset

continuum-migration-demo:
	uv run --locked sloforge continuum migrate --mode pre-copy --seed 331 --output artifacts/continuum/migration-demo --reset
	uv run --locked sloforge continuum migration verify --artifact artifacts/continuum/migration-demo/flagship.json --output artifacts/continuum/migration-demo/verification.json

continuum-fault-demo:
	uv run --locked sloforge continuum chaos --scenario scenarios/continuum/failure/destination-crash-before-commit.yaml --output artifacts/continuum/fault-demo --reset

continuum-fork-demo:
	uv run --locked sloforge continuum migrate --mode pre-copy --seed 347 --output artifacts/continuum/fork-demo --reset
	uv run --locked sloforge continuum fork --artifact artifacts/continuum/fork-demo/flagship.json --output artifacts/continuum/fork-demo/fork.json

continuum-compatibility-demo:
	uv run --locked sloforge continuum migrate --mode pre-copy --seed 359 --output artifacts/continuum/compatibility-demo --reset
	uv run --locked sloforge continuum compatibility --artifact artifacts/continuum/compatibility-demo/flagship.json --output artifacts/continuum/compatibility-demo/compatibility.json

continuum-benchmark-cpu:
	uv run --locked python -m sloforge.continuum.benchmarking --output artifacts/continuum/evaluation --seeds 101,202,303,404,505 --git-commit "$$(git rev-parse HEAD)" --initial-output-tokens 16 --delta-rounds 3,2 --resumed-tokens 3 --converter-repetitions 5 --reset
	cp artifacts/continuum/evaluation/reports/continuum-evaluation.md reports/continuum-evaluation.md
	cp artifacts/continuum/evaluation/reports/continuum-evaluation.html reports/continuum-evaluation.html
	cp artifacts/continuum/evaluation/reports/continuum-compatibility.md reports/continuum-compatibility.md
	cp artifacts/continuum/evaluation/reports/continuum-fault-tolerance.md reports/continuum-fault-tolerance.md
	cp artifacts/continuum/evaluation/reports/continuum-runtime-adapters.md reports/continuum-runtime-adapters.md

continuum-benchmark-gpu:
	./tools/continuum-gpu-benchmark.sh

continuum-docker-smoke:
	./tools/continuum-docker-smoke.sh

continuum-clean-room-test:
	./tools/clean-room-continuum.sh

helix-check:
	uv run --locked ruff format --check python/sloforge/helix python/sloforge/cli/helix.py tests/python/test_helix*.py
	uv run --locked ruff check python/sloforge/helix python/sloforge/cli/helix.py tests/python/test_helix*.py
	uv run --locked mypy python/sloforge/helix python/sloforge/cli/helix.py
	uv run --locked pytest -q tests/python/test_helix*.py
	cargo fmt --all --check
	$(CARGO_BOUNDED_ENV) cargo clippy -p sloforge-helix-ir --all-targets --all-features --locked -- -D warnings
	$(CARGO_BOUNDED_ENV) cargo test -p sloforge-helix-ir --all-features --locked

helix-demo:
	uv run --locked sloforge helix demo --seed 41 --output artifacts/helix/demo/seed-41 --replace

helix-branch-demo:
	uv run --locked pytest -q tests/python/test_helix_capture.py tests/python/test_helix_branching.py tests/python/test_helix_environment_backend.py tests/python/test_helix_rollout.py

helix-replay-demo:
	uv run --locked pytest -q tests/python/test_helix_replay.py tests/python/test_helix_demo.py

helix-training-demo:
	uv run --locked pytest -q tests/python/test_helix_dataset.py tests/python/test_helix_training.py tests/python/test_helix_training_algorithm_campaign.py tests/python/test_helix_continual_learning_campaign.py tests/python/test_helix_demo.py

helix-resource-demo:
	uv run --locked sloforge helix scheduler simulate --workload scenarios/helix/resource/cpu-learning-aware.json --policy helix_value_aware --output artifacts/helix/resource-demo
	uv run --locked pytest -q tests/python/test_helix_preservation_evaluation.py tests/python/test_helix_experience_selection_evaluation.py

helix-promotion-demo:
	uv run --locked pytest -q tests/python/test_helix_promotion.py tests/python/test_helix_learning_transaction.py tests/python/test_helix_demo.py

helix-fault-demo:
	uv run --locked pytest -q tests/python/test_helix_faults*.py tests/python/test_helix_effect_ledger.py tests/python/test_helix_promotion.py tests/python/test_helix_scheduler.py
	uv run --locked sloforge helix fault run --matrix scenarios/helix/faults/cpu-matrix.json --output artifacts/helix/faults/cpu-matrix

helix-evaluation:
	uv run --locked sloforge helix evaluate --output artifacts/helix/evaluation/reference --reports reports --seeds 41,73,113 --replace

helix-docker-smoke:
	./tools/helix-docker-smoke.sh

helix-clean-room-test:
	./tools/clean-room-helix.sh

branchfabric-trace-check:
	uv run --locked ruff format --check python/sloforge/helix/characterization python/sloforge/cli/helix.py tests/python/test_branchfabric*.py tests/python/test_helix_cli.py
	uv run --locked ruff check python/sloforge/helix/characterization python/sloforge/cli/helix.py tests/python/test_branchfabric*.py tests/python/test_helix_cli.py
	uv run --locked mypy python/sloforge/helix/characterization python/sloforge/cli/helix.py
	uv run --locked pytest -q tests/python/test_branchfabric*.py tests/python/test_helix_cli.py
	cargo fmt --all --check
	$(CARGO_BOUNDED_ENV) cargo test -p sloforge-helix-ir --test branchfabric_trace --locked

branchfabric-characterization-cpu:
	uv run --locked sloforge helix characterize run \
		--matrix benchmarks/branchfabric/characterization.yaml \
		--output artifacts/branchfabric/characterization/cpu-reference \
		--hardware cpu --seed 20260809 --max-experiments 100000 --timeout-seconds 300 \
		--replace

branchfabric-characterization-gpu:
	@set -euo pipefail; \
	status_path="artifacts/branchfabric/characterization/gpu-status.json"; \
	revision="$$(git rev-parse HEAD)"; \
	record_status() { \
		SLOFORGE_BRANCHFABRIC_STATUS_PATH="$$status_path" \
		SLOFORGE_BRANCHFABRIC_STATUS="$$1" \
		SLOFORGE_BRANCHFABRIC_STATUS_REASON="$$2" \
		SLOFORGE_BRANCHFABRIC_REVISION="$$revision" \
		SLOFORGE_BRANCHFABRIC_GPU_RESULTS_CLAIMED="$${3:-0}" \
		uv run --locked python -c 'import json, os, pathlib; path = pathlib.Path(os.environ["SLOFORGE_BRANCHFABRIC_STATUS_PATH"]); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps({"schema_version": "sloforge.branchfabric.hardware-status/v1", "status": os.environ["SLOFORGE_BRANCHFABRIC_STATUS"], "reason": os.environ["SLOFORGE_BRANCHFABRIC_STATUS_REASON"], "revision": os.environ["SLOFORGE_BRANCHFABRIC_REVISION"], "gpu_results_claimed": os.environ["SLOFORGE_BRANCHFABRIC_GPU_RESULTS_CLAIMED"] == "1", "paid_resources_created": False, "budget_env_present": bool(os.getenv("SLOFORGE_BRANCHFABRIC_CHARACTERIZATION_GPU_BUDGET_USD"))}, indent=2, sort_keys=True) + "\n")'; \
	}; \
	if test "$${SLOFORGE_BRANCHFABRIC_CHARACTERIZATION_ALLOW_GPU:-0}" != "1"; then \
		record_status unexercised "explicit local GPU opt-in is disabled; no paid resources were created"; \
		echo "BranchFabric GPU characterization unexercised: set SLOFORGE_BRANCHFABRIC_CHARACTERIZATION_ALLOW_GPU=1 to use accessible local hardware"; \
	elif ! command -v nvidia-smi >/dev/null 2>&1; then \
		record_status unavailable "nvidia-smi is unavailable; no compatible NVIDIA device was measured"; \
		echo "BranchFabric GPU characterization unavailable: no supported NVIDIA device; no hardware-backed result is claimed"; \
	else \
		record_status unavailable "compatible local NVIDIA may exist, but the GPU execution stage is not implemented; no hardware-backed result is claimed"; \
		echo "BranchFabric GPU characterization unavailable: GPU execution stage is not implemented; no result is claimed"; \
	fi

branchfabric-characterization:
	$(MAKE) branchfabric-characterization-cpu
	$(MAKE) branchfabric-characterization-gpu

branchfabric-requirements:
	uv run --locked sloforge helix characterize requirements \
		--run artifacts/branchfabric/characterization/cpu-reference \
		--output artifacts/branchfabric/requirements \
		--replace

branchfabric-report:
	uv run --locked sloforge helix characterize report \
		--run artifacts/branchfabric/characterization/cpu-reference \
		--output reports/branchfabric-characterization \
		--replace

branchfabric-clean-room-test:
	@set -euo pipefail; \
	repository_root="$$(git rev-parse --show-toplevel)"; \
	revision="$$(git -C "$$repository_root" rev-parse HEAD)"; \
	clean_root="$$(mktemp -d "$${TMPDIR:-/tmp}/sloforge-branchfabric-clean.XXXXXX")"; \
	cleanup() { rm -rf -- "$$clean_root"; }; \
	trap cleanup EXIT INT TERM; \
	git -C "$$repository_root" archive "$$revision" | tar -x -C "$$clean_root"; \
	printf '%s\n' "$$revision" > "$$clean_root/.sloforge-source-commit"; \
	unset PYTHONPATH PYTHONHOME VIRTUAL_ENV UV_PROJECT UV_PROJECT_ENVIRONMENT UV_WORKING_DIR; \
	unset CARGO_BUILD_TARGET RUSTC_WRAPPER RUSTFLAGS CARGO_ENCODED_RUSTFLAGS MAKEFLAGS MFLAGS; \
	unset SLOFORGE_BRANCHFABRIC_CHARACTERIZATION_GPU_BUDGET_USD; \
	export UV_PROJECT_ENVIRONMENT="$$clean_root/.venv"; \
	export CARGO_TARGET_DIR="$$clean_root/target"; \
	export SLOFORGE_BRANCHFABRIC_CHARACTERIZATION_ALLOW_GPU=0; \
	cd "$$clean_root"; \
	$(MAKE) bootstrap; \
	$(MAKE) branchfabric-trace-check; \
	$(MAKE) branchfabric-characterization-cpu; \
	$(MAKE) branchfabric-requirements; \
	$(MAKE) branchfabric-report; \
	$(MAKE) package

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
