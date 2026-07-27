# BranchFabric prior-negative execution replication

## Verdict

The prior negative hardware result is preserved. The tracked evidence exactly
reproduces the fanout-four sharing, immediate divergence, Continuum operation
multiset and logical COW bytes, zero multicast opportunity, selected software
baseline medians, canonical v5 instrumentation overhead, and every published
Amdahl calculation. It does not justify BranchFabric hardware.

One publication-integrity defect was found in the sealed baseline: two of the
three raw traces cited by the aggregate Helix Amdahl rows were absent from the
tracked tree and designated corpus. A read-only follow-up found both as
user-owned, untracked files in the main workspace. Their hashes and
independently derived inputs exactly close the published Helix aggregates.
The execution lead subsequently published those exact traces in commit
`82f386c` without changing the sealed prior 40-file corpus. Thus baseline
clean-checkout closure failed, while the final execution tree closes the raw
derivation. This defect does not make the hardware case stronger.

Machine-readable result:
`artifacts/branchfabric/execution/replication/prior-negative-replication.json`.
No raw artifact, source measurement, gate logic, or source performance result
was changed.

## Audit boundary and evidence classes

The audit started at commit
`46955be24d49af7090429444a0ef68f9a5695283`. The following distinctions are
mandatory when interpreting the result:

- `SYNTHETIC/HARDWARE_BACKED_REAL_HOST_TIMING` means synthetic workload data
  timed on the Apple host CPU. It is not a real model, GPU, HBM, PCIe, NIC,
  NVLink, RDMA, FPGA, or DPU measurement.
- The 75% model-state result is content-addressed accounting for tiny simulated
  state. It is not physical allocator, RSS, GPU-page, or HBM evidence.
- Continuum's 512-byte COW result is logical page-refcount attribution. It is
  not an observed OS or GPU page fault.
- Three network sends use `SIMULATED_HARDWARE`; the one host-memory send uses
  real host timing. Simulated-network timing is not hardware evidence.
- The software baselines and overhead campaign are controlled host
  microbenchmarks over synthetic payloads/workflow semantics.
- The Amdahl results are non-additive lifecycle-window sensitivities. They are
  not bounds for branch readiness, migration, a causal capacity-reclamation
  transaction, rollout throughput, or a Helix learning transaction.

## Source hashes

| Role | SHA-256 | Tracked source |
| --- | --- | --- |
| Corpus index | `bfdd1bfa2435dbc2613123e1dd8746b6b5b3e4b2b7f74f16328d6aa420f1a6e8` | `artifacts/branchfabric/CORPUS_MANIFEST.json` |
| Raw Helix lifecycle | `c98aee00cb9f8304ae2d4955c3c9ddb2ff702fc58f77946eb3b2107290afecc5` | `artifacts/branchfabric/characterization/cpu-reference/attempts/vertical_trace/attempt-000/vertical-trace/raw-lifecycle-events.json` |
| Helix sharing result | `e63b8e1f6ef2a57999fffba6a5437b5131ae9008bf738cd50f119199ed34dd71` | `artifacts/branchfabric/characterization/cpu-reference/attempts/vertical_trace/attempt-000/vertical-trace/sharing-analysis.json` |
| Helix state trace | `fb2be4a476d0508e737f2ed568d4e4cd06ed64d5289d4ba2f16a18f86598f4eb` | `artifacts/branchfabric/characterization/cpu-reference/attempts/vertical_trace/attempt-000/vertical-trace/state-operation-trace-v1.jsonl` |
| Workload analysis | `aed70e3651dad3dafbf1f4427c422ebef2e759a93711853cd8af02f1761c7c4b` | `artifacts/branchfabric/analysis/workload/cpu-reference-final.json` |
| Continuum state trace | `aa8c884f7d1e53ef824ceefc74528ea058c02de31ea6ab86db0d1271bd80fa7b` | `artifacts/branchfabric/characterization/cpu-reference/attempts/continuum_state/attempt-000/continuum-trace/state-operation-trace-v1.jsonl` |
| Continuum sharing result | `bf4565b2db19edfdf1a9cee7a3f41810d23885fccab910b515314ab4aece6a9e` | `artifacts/branchfabric/characterization/cpu-reference/attempts/continuum_state/attempt-000/continuum-trace/sharing-analysis.json` |
| Transport analysis | `7760078aa431e8822225e6805e58be6862b2a47f4c1ee0f60e71983b345e3e80` | `artifacts/branchfabric/analysis/transport/cpu-reference-v3.json` |
| Software baselines | `017b7385ba31cf913582f85f79b20e71976c3e193ab1524c844c4b434621bb9b` | `artifacts/branchfabric/software-baselines/seed-20260809/software-baselines.json` |
| Overhead raw samples | `492e90f1ea0ca35f4db4e664efacd06f9c08e20dad7f6967982a2eedd876c2df` | `artifacts/branchfabric/overhead/helix-cpu-3seed-2rep-v5-canonical/overhead-raw-samples.json` |
| Overhead analysis | `137b95696861539271d3a26299c82f99a4648a552dce04897808463744da46f2` | `artifacts/branchfabric/overhead/helix-cpu-3seed-2rep-v5-canonical/instrumentation-overhead.json` |
| Amdahl analysis | `a3012b3532c50ffb792ae573f8ece0ff6b6dcc77201055c73b52817dc72b62f4` | `artifacts/branchfabric/analysis/amdahl/cpu-reference-v3-windowed/amdahl-analysis.json` |

The four exact trajectory hashes are, in branch order:

- `alternative-tool`: `fdea44e301c279ad372fb2086c2221b5b4c0de429bf1faaf7fd3c00abd892cf8`
- `different-rng`: `5d8d2e36f651792240340fd33dbac480c71e442629af0b353585f5b51463c803`
- `forced-alternative`: `09441a4df75294e54d90ef7e4825f30e0536545ca4a003d2b172d1146ea983e8`
- `verifier-assisted`: `1470269a0f985ede93fc0f7bd9565abe225c70feb68b13a742109548506a2dc8`

## Recomputed findings

### Corpus integrity

All 40 designated entries match both recorded SHA-256 and byte length. Their
sizes sum to 8,603,992 bytes. Recomputing
`sha256(sorted path NUL sha256 NUL size newline)` yields
`9e000d81afc102c5a05666d6fb48859cb4e136a72ad6fc59f5ce086bfaa0b5d3`,
exactly matching the manifest.

This proves integrity of the curated 40-file set, not dependency closure. The
Amdahl discrepancy below is outside that closure.

Read-only follow-up found the two omitted paths in the main workspace with
SHA-256 values
`18cf226911fb4ad7f062e4ec0d958816114237a8d16a4321e777da10e6ad230b`
and
`fa0368960a8690c728c454b997dfc18affca428e4c1ddd105841e9abbff6447e`.
Both have Git status `??`; neither appears in `git ls-files`.

### Four-sibling sharing and divergence

The four raw `STATE_FORK` observations independently sum to 10,672 logical
branch bytes and 8,852 shared logical bytes. They reference one 2,668-byte
source allocation and add zero child physical bytes. Against 10,672 naive
independent bytes, sharing efficiency is exactly
`1 - 2,668 / 10,672 = 0.75`; physical amplification is 1.0.

Every exact trajectory contains one token and one action. The token sequences
are `ast_guided`, `naive_parse`, `guarded_parse`, and `verifier_assisted`; the
action sequences have those same four values. Both first-difference scans
return index 0. This exactly reproduces the immediate-divergence negative case.

### Continuum multiset and logical COW

The 18-event canonical trace contains:

| Operation | Count |
| --- | ---: |
| `STATE_SNAPSHOT` | 2 |
| `STATE_SEND` | 4 |
| `STATE_RECEIVE` | 4 |
| `STATE_PUBLISH` | 1 |
| `STATE_FORK` | 1 |
| `STATE_COW` | 1 |
| `STATE_APPEND` | 1 |
| `STATE_DELTA` | 1 |
| `STATE_RESHARD` | 1 |
| `STATE_COMMIT` | 1 |
| `STATE_RECLAIM` | 1 |

The single COW event records 512 bytes and eight logical pages, exactly
matching the sharing artifact.

### Transfer and multicast opportunity

The trace contains four successful sends, all at fanout one. Source and logical
delivery bytes are both 8,114; receive bytes are 8,114; retry bytes are zero.
The conservative detector retains zero opportunities, four ungrouped sends,
and zero reducible bytes at minimum fanouts 2, 4, 8, 16, 32, 64, and 128.

### Strongest controlled software baselines

Each median below was recomputed from all seven non-warmup raw samples; no
outlier was removed. Each selected case is the lowest raw median in its
recorded semantic-equivalence candidate set.

| Selection | Selected case | Median |
| --- | --- | ---: |
| Transform | `transform.trusted_canonical_staged` | 5,899,000 ns |
| SHA-256 | `hash.hashlib_sha256_whole` | 79,333 ns |
| In-process transfer | `in_process_transfer.continuum_in_process.chunk_65536.concurrency_4` | 296,708 ns |
| Fanout 2 | `software_fanout.repeated_unicast.fanout_2` | 36,000 ns |
| Fanout 4 | `software_fanout.binary_tree.fanout_4` | 80,542 ns |
| Fanout 8 | `software_fanout.repeated_unicast.fanout_8` | 139,792 ns |
| Fanout 16 | `software_fanout.binary_tree.fanout_16` | 294,083 ns |
| Fanout 32 | `software_fanout.repeated_unicast.fanout_32` | 524,875 ns |
| Fanout 64 | `software_fanout.repeated_unicast.fanout_64` | 1,114,750 ns |
| Fanout 128 | `software_fanout.binary_tree.fanout_128` | 2,374,333 ns |

The staged transform beats the direct controlled path's 8,905,459 ns median.
Whole-buffer SHA-256 reports 3.304 GB/s logical throughput, and the selected
in-process transfer reports 883.5 MB/s. These are host microbenchmarks. The
software fanout cells do not turn the measured fanout-one transfer trace into a
multicast opportunity.

### Canonical v5 instrumentation overhead

The raw campaign contains three level-specific warmups and 18 measured trials:
six per level, covering seeds 41, 73, and 113 with two repetitions each.
Semantic digests match across trace levels for each seed.

| Level | Wall median | End-to-end median | CPU median |
| --- | ---: | ---: | ---: |
| Disabled | 11,344,447,395.5 ns | 11,344,447,395.5 ns | 2,162,809,000 ns |
| Minimal | 11,378,086,833 ns | 11,382,661,083 ns | 2,193,251,000 ns |
| Full | 11,389,598,208 ns | 11,395,694,895.5 ns | 2,179,386,000 ns |

Matched by seed and repetition, full minus disabled has median hot-path wall
change +37,781,250 ns, or +0.3328247547%. Including persistence, the matched
median change is +43,978,271 ns, or +0.3874507692%. Full persistence has a
6,090,458.5 ns median and a 210,867-byte median. These reproduce the final
report after rounding.

### Amdahl bounds

For every published row and acceleration, the audit recomputed:

`projected = total - primitive + primitive / acceleration`

and `speedup = total / projected`; the free case removes the primitive term.
All 32 published speedups match exactly within floating-point tolerance.

| Window | Primitive | Fraction | 2x | 5x | 10x | Free |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Continuum | Reshard | 1.669361% | 1.008417x | 1.013536x | 1.015253x | 1.016977x |
| Continuum | Delta extraction | 0.606980% | 1.003044x | 1.004880x | 1.005493x | 1.006107x |
| Continuum | Commit | 0.094814% | 1.000474x | 1.000759x | 1.000854x | 1.000949x |
| Continuum | COW | 0.021218% | 1.000106x | 1.000170x | 1.000191x | 1.000212x |
| Continuum | Reclamation | 0.019548% | 1.000098x | 1.000156x | 1.000176x | 1.000196x |
| Continuum | Transfer | 0.933986% | 1.004692x | 1.007528x | 1.008477x | 1.009428x |
| Helix | COW | 0.383958% | 1.001923x | 1.003081x | 1.003468x | 1.003854x |
| Helix | Reclamation | 0.242572% | 1.001214x | 1.001944x | 1.002188x | 1.002432x |

The Continuum trace independently yields the reported 99,756,167 ns window
and every primitive-exclusive input. The one tracked Helix trace independently
yields a 447,060,958 ns window, 1,660,459 ns COW, and 1,064,457 ns
reclamation. The published Helix rows aggregate that trace with two raw inputs
absent from the tracked baseline. The then-untracked local inputs independently
yield windows 445,814,375 ns and 468,730,875 ns; COW spans 1,778,333 ns and
1,789,208 ns; and reclamation spans 1,104,873 ns and 1,133,542 ns. Combined
with the tracked trace, totals are exactly 1,361,606,208 ns, 5,228,000 ns COW,
and 3,302,872 ns reclamation, matching both Helix rows. The largest reported
free-operation sensitivity is 1.016977x, below 1.02x. No end-to-end
target-objective bound exists.

## Discrepancy

### BF-NEG-REPL-H1 — High: two Helix Amdahl raw traces are absent from the tracked baseline

The Helix Amdahl rows cite:

- `artifacts/branchfabric/characterization/cpu-reference/attempts/instrumentation_overhead/attempt-000/helix-overhead/trials/000-full-s20260809-r0/state-operation-trace-v1.jsonl`
- `artifacts/branchfabric/characterization/cpu-reference/attempts/instrumentation_overhead/attempt-000/helix-overhead/trials/004-full-s20260809-r0/state-operation-trace-v1.jsonl`

Neither path exists in the tracked tree, and neither is bound by the 40-entry
corpus manifest. Thus only one of three Helix samples is derivable in a clean
checkout. The two files do exist untracked in the current main workspace. A
read-only recomputation shows that they exactly close operation counts,
primitive times, and total times. The disposition is therefore: numerically
closed from local untracked evidence, not closed for the sealed published
baseline. The execution lead fixed final-tree availability by publishing the
same validated traces in commit `82f386c`; the sealed prior corpus was not
rewritten.

This is high severity for the evidence-reproduction contract. It does not
weaken the no-build decision: mandatory real workload/hardware evidence and
end-to-end relevance remain absent, while the reported aggregate is itself
strongly negative.

No other requested value discrepancy was found.

## Preliminary gate semantic challenge

The main workspace's preliminary gate was independently inspected at these
snapshot hashes:

| Gate artifact | SHA-256 |
| --- | --- |
| `python/sloforge/helix/characterization/gates.py` | `dd723b2452162882dec222c91bb9c4b6d45880d31fd197f1a61e351d1cce90b6` |
| `tests/python/test_branchfabric_gates.py` | `557cae7a951741a61d1edbb1af0a4a430669b18cf3ee62d9df7200fde5640dcd` |
| `artifacts/branchfabric/gates/branchfabric_gate_input.json` | `7a1f7a84a5de9518f702f71112ad8ccdc75e25378fd43f03e4d2a606cd441fe6` |
| `artifacts/branchfabric/gates/branchfabric_gate_result.json` | `85f7b67842a1c4be940ced1e2027f745ec64470652c206f06906130d3e24cfca` |
| `BRANCHFABRIC_GATE_REPORT.md` | `cef2c579b5b05a4b490bfa0b97b84544901a57f2e0c8ba1c10e3e1c09fb9f0fb` |

The current semantics satisfy all three requested fail-closed checks:

- `CPU_REFERENCE_MODEL_STATE` is not `TARGET_HARDWARE_REAL`; even otherwise
  passing candidate fields cannot satisfy mandatory real evidence.
- Every current system-headroom lower-bound field is null. The isolated
  free-operation lifecycle bounds remain only relevance context and cannot be
  counted as confidence-backed headroom.
- `functional_model_or_cycle_simulator_allowed` is assigned the same
  `bool(passing_candidates)` condition as hardware implementation. The current
  `FAIL_NO_BUILD` result sets both false and requires `TERMINATE_HARDWARE_PATH`.

One medium future-hardening finding remains. `EvidenceReference` has no binding
to a workload class, seed, or derived metric, and `HeadroomEvidence` accepts
bare numeric fields. The evaluator therefore verifies artifact hashes and
threshold arithmetic, but not that a claimed future lower confidence bound was
actually derived from the cited end-to-end raw samples rather than an isolated
projection. The current preliminary result is unaffected; a future PASS should
bind each critical-path/headroom statistic to raw sample selectors, confidence
method, seeds, workload classes, and target-hardware timing provenance.

## Exact commands

Baseline and source status:

```sh
git rev-parse HEAD
git status --short --branch
git ls-files 'artifacts/branchfabric/**/state-operation-trace-v1.jsonl' | grep '000-full-s20260809\|004-full-s20260809'
```

Corpus verification:

```sh
python3 - <<'PY'
import hashlib, json
from pathlib import Path
m=json.loads(Path('artifacts/branchfabric/CORPUS_MANIFEST.json').read_text())
parts=[]
for a in sorted(m['artifacts'], key=lambda a:a['path']):
    p=Path(a['path']); h=hashlib.sha256(p.read_bytes()).hexdigest()
    assert h == a['sha256']; assert p.stat().st_size == a['size_bytes']
    parts.append(f"{a['path']}\0{h}\0{a['size_bytes']}\n".encode())
digest=hashlib.sha256(b''.join(parts)).hexdigest()
print(len(parts), sum(a['size_bytes'] for a in m['artifacts']), digest)
assert digest == m['corpus_sha256']
PY
```

Four-sibling accounting:

```sh
jq '[.state_events[] | select(.operation_type=="STATE_FORK")] | . as $f | {branch_count:length, logical_branch_bytes:(map(.logical_bytes)|add), shared_logical_bytes:(map(.shared_logical_bytes)|add), physical_allocated_bytes:($f[0].source_physical_bytes+(map(.physical_bytes)|add)), naive_independent_bytes:(map(.naive_independent_bytes)|add)} | . + {sharing_efficiency:(1-(.physical_allocated_bytes/.naive_independent_bytes))}' artifacts/branchfabric/characterization/cpu-reference/attempts/vertical_trace/attempt-000/vertical-trace/raw-lifecycle-events.json
```

Exact divergence inputs:

```sh
jq -s 'map({branch_id,tokens:[.tokens[].token],actions:[.actions[].action]})' artifacts/branchfabric/characterization/cpu-reference/attempts/vertical_trace/attempt-000/vertical-trace/helix-demo/trajectories/*.json
```

Continuum multiset and logical COW:

```sh
jq -s 'group_by(.operation_type)|map({key:.[0].operation_type,value:length})|from_entries' artifacts/branchfabric/characterization/cpu-reference/attempts/continuum_state/attempt-000/continuum-trace/state-operation-trace-v1.jsonl
jq -s '[.[]|select(.operation_type=="STATE_COW")]|{cow_bytes:(map(.bytes)|add),cow_page_count:(map(.attributes.cow_pages)|add)}' artifacts/branchfabric/characterization/cpu-reference/attempts/continuum_state/attempt-000/continuum-trace/state-operation-trace-v1.jsonl
```

Transport fanout and bytes:

```sh
jq -s '[.[]|select((.operation_type=="STATE_SEND" or .operation_type=="STATE_MULTICAST") and .result=="success")]|{send_event_count:length,fanouts:map(.fanout),source_payload_bytes:(map(.bytes)|add),logical_delivery_bytes:(map(.bytes*.fanout)|add),explicit_fanout_opportunities:(map(select(.fanout>=2))|length)}' artifacts/branchfabric/characterization/cpu-reference/attempts/continuum_state/attempt-000/continuum-trace/state-operation-trace-v1.jsonl
jq '.multicast|{opportunity_count,ungrouped_send_event_count,by_minimum_fanout}' artifacts/branchfabric/analysis/transport/cpu-reference-v3.json
```

Selected software medians:

```sh
python3 - <<'PY'
import json, statistics
from collections import defaultdict
from pathlib import Path
p=Path('artifacts/branchfabric/software-baselines/seed-20260809/software-baselines.json')
d=json.loads(p.read_text()); samples=defaultdict(list)
for s in d['raw_samples']:
    if not s['warmup']: samples[s['case_id']].append(s['duration_ns'])
med={k:statistics.median(v) for k,v in samples.items()}
for choice in d['selected_baselines']:
    winner=min(choice['candidate_case_ids'], key=lambda c:(med[c],c))
    print(choice['selection_id'], winner, med[winner], len(samples[winner]))
    assert winner == choice['case_id']
PY
```

Canonical overhead recomputation:

```sh
python3 - <<'PY'
import json, statistics
from pathlib import Path
p=Path('artifacts/branchfabric/overhead/helix-cpu-3seed-2rep-v5-canonical/overhead-raw-samples.json')
d=json.loads(p.read_text()); measured=[t for t in d['trials'] if not t['warmup']]
levels={x:sorted((t for t in measured if t['trace_level']==x),key=lambda t:(t['seed'],t['repetition'])) for x in ('disabled','minimal','full')}
for level,ts in levels.items(): print(level, statistics.median(t['wall_time_ns'] for t in ts), statistics.median(t['end_to_end_wall_time_ns'] for t in ts), statistics.median(t['cpu_time_ns'] for t in ts))
base={(t['seed'],t['repetition']):t for t in levels['disabled']}; full={(t['seed'],t['repetition']):t for t in levels['full']}; keys=sorted(base)
wall=[full[k]['wall_time_ns']-base[k]['wall_time_ns'] for k in keys]; rel=[d/base[k]['wall_time_ns'] for k,d in zip(keys,wall)]
e2e=[full[k]['end_to_end_wall_time_ns']-base[k]['end_to_end_wall_time_ns'] for k in keys]; e2erel=[d/base[k]['end_to_end_wall_time_ns'] for k,d in zip(keys,e2e)]
print('full-minus-disabled',statistics.median(wall),statistics.median(rel),statistics.median(e2e),statistics.median(e2erel))
PY
```

Amdahl formula and raw-reference verification:

```sh
python3 - <<'PY'
import json, math
from pathlib import Path
p=Path('artifacts/branchfabric/analysis/amdahl/cpu-reference-v3-windowed/amdahl-analysis.json')
d=json.loads(p.read_text())
for r in d['results']:
    assert math.isclose(r['critical_path_fraction'],r['primitive_exclusive_duration_ns']/r['total_duration_ns'])
    for b in r['bounds']:
        factor=b['primitive_acceleration']; projected=r['total_duration_ns']-r['primitive_exclusive_duration_ns']+(0 if factor is None else r['primitive_exclusive_duration_ns']/factor)
        assert math.isclose(projected,b['projected_duration_ns']); assert math.isclose(r['total_duration_ns']/projected,b['projected_speedup'])
    print(r['objective'],r['primitive'],[(b['scenario'],b['projected_speedup']) for b in r['bounds']],[(x,Path(x).is_file()) for x in r['artifact_references']])
PY
```

Untracked Helix-input follow-up, without copying the files:

```sh
for rel in artifacts/branchfabric/characterization/cpu-reference/attempts/instrumentation_overhead/attempt-000/helix-overhead/trials/000-full-s20260809-r0/state-operation-trace-v1.jsonl artifacts/branchfabric/characterization/cpu-reference/attempts/instrumentation_overhead/attempt-000/helix-overhead/trials/004-full-s20260809-r0/state-operation-trace-v1.jsonl; do f="/Users/rishivinodkumar/sloforge/$rel"; shasum -a 256 "$f"; git -C /Users/rishivinodkumar/sloforge status --short -- "$rel"; git -C /Users/rishivinodkumar/sloforge ls-files --error-unmatch "$rel"; done
python3 - <<'PY'
import json
from pathlib import Path
root=Path('/Users/rishivinodkumar/sloforge')
paths=[root/'artifacts/branchfabric/characterization/cpu-reference/attempts/instrumentation_overhead/attempt-000/helix-overhead/trials/000-full-s20260809-r0/state-operation-trace-v1.jsonl',root/'artifacts/branchfabric/characterization/cpu-reference/attempts/instrumentation_overhead/attempt-000/helix-overhead/trials/004-full-s20260809-r0/state-operation-trace-v1.jsonl',Path('artifacts/branchfabric/characterization/cpu-reference/attempts/vertical_trace/attempt-000/vertical-trace/state-operation-trace-v1.jsonl')]
totals={'total':0,'cow':0,'reclamation':0}
for path in paths:
    events=[json.loads(line) for line in path.read_text().splitlines()]
    totals['total']+=max(e['normalized_timestamp_ns']+e['duration_ns'] for e in events)-min(e['normalized_timestamp_ns'] for e in events)
    totals['cow']+=sum(e['operation_latency_ns'] for e in events if e['operation_type']=='STATE_COW')
    totals['reclamation']+=sum(e['operation_latency_ns'] for e in events if e['operation_type']=='STATE_FREE')
print(totals); assert totals=={'total':1361606208,'cow':5228000,'reclamation':3302872}
PY
```

Preliminary gate snapshot and semantics:

```sh
shasum -a 256 /Users/rishivinodkumar/sloforge/python/sloforge/helix/characterization/gates.py /Users/rishivinodkumar/sloforge/tests/python/test_branchfabric_gates.py /Users/rishivinodkumar/sloforge/artifacts/branchfabric/gates/branchfabric_gate_input.json /Users/rishivinodkumar/sloforge/artifacts/branchfabric/gates/branchfabric_gate_result.json /Users/rishivinodkumar/sloforge/BRANCHFABRIC_GATE_REPORT.md
jq '{outcome,passing_candidates,hardware_implementation_allowed,functional_model_or_cycle_simulator_allowed,required_action}' /Users/rishivinodkumar/sloforge/artifacts/branchfabric/gates/branchfabric_gate_result.json
uv run --locked pytest -q tests/python/test_branchfabric_gates.py
```

Focused acceptance:

```sh
uv run --locked pytest -q tests/python/test_branchfabric_amdahl.py tests/python/test_branchfabric_workload_analysis.py tests/python/test_branchfabric_transport_analysis.py tests/python/test_branchfabric_software_baselines.py tests/python/test_branchfabric_overhead.py tests/python/test_branchfabric_trace_io.py tests/python/test_branchfabric_trace_models.py
cargo test -p sloforge-helix-ir --test branchfabric_trace
jq empty artifacts/branchfabric/execution/replication/prior-negative-replication.json
git diff --check
```

The focused Python suite passed with 37 tests, the Rust trace suite passed
with 5 tests, and the preliminary gate suite passed with 5 tests. JSON parsing,
machine assertions, and `git diff --check` also passed.

The repository-wide `make check` did not pass cleanly. The first invocation
resolved `pytest` from the main workspace virtual environment and reported 3
failures, 1,245 passes, and 6 skips. After creating and prioritizing the
worktree-local locked environment, the two environment-contaminated failures
passed. The remaining failure is the unrelated
`test_actual_hidden_black_box_reward_rejects_plausible_wrong_patch` assertion:
it rejects a valid verifier SHA whenever its hexadecimal text happens to
contain the substring `300`; the observed SHA did. No owned BranchFabric file
was implicated. This audit does not hide or relabel that repository-level
failure.

## Adversarial hardware-justification answer

**Would this hardware have been selected from the measured workload if the
BranchFabric name and prior proposal did not already exist?**

No.

The workload-selected decision would be to retain optimized CPU software and
collect representative model/GPU/network/transaction evidence. The only
branch group has four siblings, state is tiny and simulated, divergence is
immediate, every transfer has fanout one, metadata demand is unmeasured, queue
concurrency is serialized, and no end-to-end Helix objective has a measured
hardware headroom bound. The maximum lifecycle-window free-operation
sensitivity is only 1.016977x. No neutral architect would select an FPGA, DPU,
multicast engine, hardware page table, or shared-root HBM/CXL store from this
evidence.
