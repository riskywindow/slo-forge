# Genesis CPU demonstration script

This walkthrough exercises the implemented offline CPU vertical slice. It does not use a GPU, a
cloud resource, an external model service, or a paid synthesis API. All synthesis, adversarial, and
benchmark schedules take explicit seeds.

## Three-minute artifact-backed path

From a bootstrapped checkout, run:

```bash
make genesis-demo
make synthbench-smoke

python - <<'PY'
import json
from pathlib import Path

demo = json.loads(Path("artifacts/genesis/demo/GENESIS_DEMO_REPORT.json").read_text())
assert demo["runtime_differential_passed"] is True
assert demo["cross_layer_accepted"] is True
assert demo["rejected_candidate_ids"]
assert demo["minimized_counterexample_ids"]
assert demo["capsule_promotion_eligible"] is True
assert demo["evolution_promoted"] is True
assert demo["active_stream_preserved"] is True
assert demo["hardware_backed"] is False

bench = json.loads(Path("artifacts/synthbench/smoke/summary.json").read_text())
raw_report = json.loads(Path("artifacts/synthbench/smoke/run/report.json").read_text())
assert bench["hardware_backed"] is False
assert raw_report["report_source"] == "derived_from_raw_samples"
print(json.dumps({"genesis": demo, "synthbench": bench}, indent=2, sort_keys=True))
PY
```

`make genesis-demo` creates the inspection, baseline genome/runtime, CEGIS event log, rejected and
accepted candidates, two independently built capsules, red-team corpus, CPU kernel experiment,
champion/challenger timeline, and final machine-readable report under `artifacts/genesis/demo/`.
`make synthbench-smoke` creates seeded grammar tasks, evaluator-only hidden commitments, raw CPU
timings, integrity findings, and an artifact-derived report under `artifacts/synthbench/smoke/`.

The headline artifacts are:

- `artifacts/genesis/demo/inspection/inspection.json`
- `artifacts/genesis/demo/run/inference_genome.json`
- `artifacts/genesis/demo/run/generated_runtime/`
- `artifacts/genesis/demo/run/synthesis/cegis/`
- `artifacts/genesis/demo/capsule/manifests/`
- `artifacts/genesis/demo/evolution/challenger-capsule/manifests/`
- `artifacts/genesis/demo/evolution/timeline.json`
- `artifacts/genesis/demo/kernel/lab/`
- `artifacts/genesis/demo/GENESIS_DEMO_REPORT.json`
- `artifacts/synthbench/smoke/summary.json`
- `artifacts/synthbench/smoke/run/report.json`

The demo's performance claim is a deterministic service-model simulation. Local CPU timing is
retained for the focused kernel experiment, but it does not become a serving speedup claim. The
flagship report deliberately records `hardware_backed=false`.

## Inspect, compile, and synthesize manually

The manual path makes the compiler boundary visible. The output root must not already exist because
Genesis refuses to overwrite evidence.

```bash
set -euo pipefail

SEED=73129
DEMO_ROOT="artifacts/genesis/manual-${SEED}"
PACKAGE="models/reference_tasks/hybrid_decoder"
test ! -e "${DEMO_ROOT}"
mkdir -p "${DEMO_ROOT}/inputs"

python - "${DEMO_ROOT}/inputs/hardware.json" <<'PY'
import hashlib
import json
import platform
import sys
from pathlib import Path

identity = {
    "system": platform.system(),
    "machine": platform.machine(),
    "processor": platform.processor(),
}
fingerprint = hashlib.sha256(
    json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "schema_version": "1.0.0",
            "architecture": "cpu",
            "memory_bytes": 8 * 1024**3,
            "measured_identity": identity,
            "measured_fingerprint": fingerprint,
        },
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

uv run --locked sloforge genesis inspect \
  --reference "${PACKAGE}" \
  --seed "${SEED}" \
  --output "${DEMO_ROOT}/inspection"

uv run --locked sloforge genesis initialize \
  --inspection "${DEMO_ROOT}/inspection" \
  --workload "${PACKAGE}/search_samples.jsonl" \
  --hardware "${DEMO_ROOT}/inputs/hardware.json" \
  --seed "${SEED}" \
  --output "${DEMO_ROOT}/run"

uv run --locked sloforge genesis synthesize \
  --run "${DEMO_ROOT}/run" \
  --budget-usd 0 \
  --seed "${SEED}"
```

The AST frontend records a typed call inventory, persistent state, aliases, control flow, and contract
obligations without importing the model. It does **not** invent SSA input/output bindings or tensor
metadata. Consequently, the emitted `TensorGenome` marks the recovered call inventory as
`unresolved_static_call_inventory`, retains every unresolved operation in its extension payload, and
enables no algebraic rewrite over those unresolved calls. The generated conservative runtime uses
the declared reference entry points; it is not evidence that the static call inventory is an
executable algebraic graph.

Extract the candidate identities and independently replay the scoped cancellation verifier:

```bash
ACCEPTED_ID=$(python - "${DEMO_ROOT}/run/synthesis/result.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["accepted_candidate_id"])
PY
)

REJECTED_ID=$(python - "${DEMO_ROOT}/run/synthesis/result.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["rejected_candidate_ids"][0])
PY
)

uv run --locked sloforge genesis verify \
  --candidate "${DEMO_ROOT}/run/candidates/${ACCEPTED_ID}"

if uv run --locked sloforge genesis verify \
  --candidate "${DEMO_ROOT}/run/candidates/${REJECTED_ID}"; then
  echo "unsafe candidate unexpectedly passed" >&2
  exit 1
else
  echo "unsafe candidate rejected as expected"
fi
```

The verifier actually executes the unsafe request schedule, observes an emission after cancellation,
and minimizes the schedule to `admit, cancel, emit`. The learned
`cancel_check_before_emit == true` precondition suppresses a repeated family member before Genesis
selects the corrected request/serving candidate.

## Validate with an external trust context

Capsule integrity is not identity authentication. The operator must pin the expected capsule digest
and trusted evidence anchors outside the untrusted capsule tree. For the local demo, the trusted
builder emits a convenience context; copy it into an operator-controlled location before treating the
capsule directory as hostile:

```bash
CAPSULE_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)

uv run --locked sloforge genesis capsule build \
  --candidate "${DEMO_ROOT}/run/candidates/${ACCEPTED_ID}" \
  --timestamp "${CAPSULE_TIME}" \
  --output "${DEMO_ROOT}/capsule" \
  > "${DEMO_ROOT}/capsule-build.json"

mkdir -p "${DEMO_ROOT}/operator-trust"
cp "${DEMO_ROOT}/capsule/validation_context.json" \
  "${DEMO_ROOT}/operator-trust/validation_context.json"

EXPECTED_CAPSULE_DIGEST=$(python - "${DEMO_ROOT}/capsule-build.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["capsule_digest"])
PY
)

uv run --locked sloforge genesis capsule validate \
  "${DEMO_ROOT}/capsule" \
  --context "${DEMO_ROOT}/operator-trust/validation_context.json" \
  --expected-digest "${EXPECTED_CAPSULE_DIGEST}"
```

The CLI rejects a context located inside `${DEMO_ROOT}/capsule`. The external context binds the source,
tokenizer, workload, hardware, dependency lock, verifier version, expected capsule digest, and each
promotion evidence record to its issuer/version and exact artifact digests. In a real deployment that
context and expected digest must arrive over an operator-controlled trust channel; copying the local
builder output is only a reproducible offline fixture.

## Evolution and evaluation scope

The flagship demo evolves between two real, separately synthesized and independently validated
capsules. It registers the second as an isolated challenger, supplies deterministic local shadow and
canary observations bound to candidate/capsule/evidence digests and controller seed, revalidates
immediately before promotion, preserves the active stream lease on the original champion, and retains
the old capsule for rollback. It then records a **simulated** fabric-degradation trigger. This is not
external traffic, physical link failure, or hardware state migration.

Run the multi-seed hypothesis report with:

```bash
make genesis-evaluation
```

The resulting `artifacts/genesis/evaluation/evaluation.json` reports H2 and H4 as not evaluated.
Lineage mechanics and a deterministic transfer demonstration exist, but the required performance
campaign for H5 remains unevaluated. H7 and H9 are only partially evaluated. No NVIDIA GPU, Linux
sandbox backend, Docker daemon, multi-node environment, or external live promotion was exercised on
the checked Darwin host.
