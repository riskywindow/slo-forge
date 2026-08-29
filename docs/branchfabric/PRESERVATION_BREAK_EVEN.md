# BranchFabric preservation break-even

Status: `NOT_MEASURED`

A measured break-even cannot be fit. The corrected targeted identity gate passed
1,152/1,152, but the one full integrated optimized-preserve retry failed during
Modal image build before the remote function entered. The precommitted experiment
ordering therefore forbids running `KILL_AND_RECOMPUTE`. The measured export/restore fixed
costs, integrated state-movement throughput, actual recomputed token count,
prefill throughput, recompute GPU-seconds, and interruption latency are absent.

The intended interpretable model remains:

```text
recompute_tokens = B(P + A)
state_bytes      = q(P + BA)

T_preserve = C_preserve + state_bytes / BW_state
T_recompute = C_recompute + recompute_tokens / R_prefill
```

where `P` is prefix length, `B` branch count, `A` private trajectory age,
`q` bytes per preserved token-equivalent, `BW_state` measured integrated state
throughput, and `R_prefill` measured recompute throughput. No coefficient or
break-even frontier is reported until both real arms exist. The analytical
133,120-token expectation is not labeled measured.

Evidence: [break-even status](../../artifacts/branchfabric/gpu-validation/experiment-004/v11-final/break-even/status.json).
