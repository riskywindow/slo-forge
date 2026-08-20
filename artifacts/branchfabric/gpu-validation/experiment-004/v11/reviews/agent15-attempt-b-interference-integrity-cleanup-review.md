# Agent 15 attempt-B interference, integrity, and cleanup review

Status: **PASS review of a terminally FAILED/INVALID experiment**.

Attempt `exp004-v11-integrated-s41-b` authenticated two distinct NVIDIA A100-SXM4-80GB devices, brought both retained engines to readiness, passed the short 12-rps and 15-rps sanity guards, and reached transaction-ready with eight rollout branches. GPU0 then hit the fail-closed bounded-backlog limit before reclamation: 65 outstanding requests against the hard ceiling of 64, comprising 16 running, 48 waiting, and one external queued request.

No reclaim-trigger barrier was written. GPU1's v11 state path waits on that immutable barrier before building the export plan or invoking optimized capture. Consequently, attempt B produced no integrated export, release, post-free ownership proof, HBM reclamation, GPU1 recovery, import, admission, continuation, StatePassRecord ledger, movement accounting, restore latency, or GPU0-during-restore interference measurement. The valid attempt-D micro result must not be presented as an attempt-B integrated result.

The full remote manifest is intact: all 460 rows and 20,098,327 bytes were downloaded and matched their declared size and SHA-256, with no missing or unexpected files. No manifest path represents reclaim, export, import, state passes, movement, ownership, restore, or admission.

Cleanup requires a deliberate distinction. The in-run controller cleanup gate failed because rollout process group 9 was still visible after SIGTERM and SIGKILL; both-worker typed teardown is not proved. The controller nevertheless observed no remaining GPU compute process. The independent provider postflight then proved the app stopped and zero tasks, containers, active apps, endpoints, reservations, child processes, and profilers. Thus final provider cleanup passes and no resource leak remains, but controller-local cleanup does not become a scientific PASS retroactively.

The ledger conservatively charges the full 1,176 GPU-seconds ($0.816144), records 16,113.513626085978 cumulative GPU-seconds, leaves 5,486.486373914022 authorized GPU-seconds, and has no active reservation. No retry or third integrated attempt is authorized.
