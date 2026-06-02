# SLOForge Fabric CPU demonstration

All values below are loaded from the manifest and raw simulator artifacts.
Synthetic hardware curves are labeled synthetic and are not GPU measurements.

## Outcome

- Physical plan: `physical-plan-75732ff902f1c340`
- Diagnosis: `rank_straggler` (0.900 confidence)
- Counterfactuals evaluated: 7
- Live Rust gateway SSE requests: 12
- Selected repair: `remove-both-faults`
- Recovery state: `COMPLETED`

| run | p95 TTFT ms | p99 TPOT ms | p95 E2E ms | SLO attained |
|---|---:|---:|---:|:---:|
| healthy | 955.883 | 1.250 | 985.883 | yes |
| degraded | 2484.092 | 5.000 | 5524.592 | no |
| restored | 955.883 | 1.250 | 985.883 | yes |

## Artifact-derived timeline

- +0.000s **SLO_REGRESSION** — p95 TTFT 2484.092 ms exceeded 1051.471 ms (`autopsy/comparison.json`)
- +0.001s **CAUSAL_DIAGNOSIS** — rank_straggler confidence=0.900 (`autopsy/diagnosis.json`)
- +0.002s **COUNTERFACTUAL_SELECTION** — evaluated 7 repairs; selected remove-both-faults (`autopsy/counterfactuals.json`)
- +1.000s **VALIDATED_IN_SIMULATION** — counterfactual simulation validation passed (`recovery/execution.json`)
- +2.000s **BUILDING_REPLACEMENT** — simulated action change_rank_placement completed (`recovery/execution.json`)
- +3.000s **SHADOWING** — replacement is ready; starting shadow traffic (`recovery/execution.json`)
- +4.000s **CANARYING** — shadow sample and SLO criteria passed (`recovery/execution.json`)
- +5.000s **PROMOTING** — canary sample and SLO criteria passed (`recovery/execution.json`)
- +6.000s **DRAINING_OLD** — traffic migrated; old workers no longer receive new requests (`recovery/execution.json`)
- +8.000s **COMPLETED** — old workers drained; recovery completed (`recovery/execution.json`)
- +9.000s **SLO_RESTORED** — p95 TTFT restored to 955.883 ms (`simulations/restored.json`)

## Artifact integrity

- `topology.json` — `d7e67e922b9da9322db38b36a6ca2d39d69d01135ea0aab29472867d0658bd36`
- `fabric-profile.json` — `ed0cab849fef11b84151d9527ee1de15790b3e5360f9c47b783fd03ef0e3eb06`
- `model-graph.json` — `1ce629c74f379507a941d6354ca73313be760bf8ff3304948aad563af787f00a`
- `physical-plan.json` — `cec561e35597a564332222cb374d9ae265312bd437c5bb8f9d17243ec029133d`
- `physical-plan-topology-unaware.json` — `78a7caf03def67d6409d713d198b00a9032accc5c76d051fc3beafb0a37e9e5e`
- `mixed-bursty.jsonl` — `99a473c138773bb5f8c447d86a9e24d6f78a3fbc874ae6f3e8dec8221d0bdc69`
- `simulations/healthy.json` — `a31a1cad1cc7a4c71ea45482fbfd41b8a9731428faf196cacaefe35241a523b9`
- `simulations/degraded.json` — `f16134495a26c91e8c79746a83e357d34389505b1a721c87efda93b7277ee786`
- `simulations/restored.json` — `3983c61ada28589dc387f4b42fb00a132450eeaca361193814e54b8dc63a684a`
- `autopsy/comparison.json` — `181e7ec9a8b8368c15a08f859b3922861489181fee4beeeb54eec29112f6082f`
- `autopsy/diagnosis.json` — `d8173a5465031087774de02290c0f69cdf8ee73d6210e1295dd07928d5bde45c`
- `autopsy/counterfactuals.json` — `f5290b9c90a15cdac945a9bc7422b1b703d70cc14e97b44805705a4f1f3634c7`
- `recovery/proposal.json` — `74d817db27fa0383a12064581aa48157fb45cc7ecea7a57f6a2efc6679e83c14`
- `recovery/execution.json` — `2a72d7dbef66365620cfa28e34193280d3e6b73fc143fb8c913794aaf235473b`
- `traces/degraded.perfetto.json` — `cab8dcfa92051dc6444d44c59fe59e925fe7a704d6959ea6b3e6e4eaff15a633`
- `traces/otel.json` — `8f590ba119b08dd05442b78009b4389289953831e7ddf4b0eb341136011b8b93`
- `metrics/degraded.prom` — `c0efe75a83826e5a7e11352e5d18041fb3e081b734c7dd7693c469f3b213077a`
- `timeline.json` — `d606825f374023921c878bed467a6f54eff62f6a019efd4d3c0f57671aef370c`
- `runtime/gateway-replay.json` — `1cdd1798af9c8d375c04c955cf05cb81105d87427eef3a2a6b9371486dcbe011`
- `runtime/gateway.prom` — `cb7cba353f69cc0a2b4644c1178c442cee4ae6253a4a635327070cb13b071085`
- `runtime/gateway.perfetto.json` — `6c2f294ffa92e0983da55a42a2bdb5ca522f056c32e867eac867e274834c17ee`
