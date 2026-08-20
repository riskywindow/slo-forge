# Agent 12 adversarial software review

Verdict: **PASS — admit exactly one integrated v11 transaction, subject to the
mandatory conditions below.**

I reviewed real-A100 attempt `exp004-v11-micro-s41-d` as an adversarial software
reviewer. I did not invoke CUDA, Modal, a GPU, or the network; I did not modify
source or frozen v10.

## What survives challenge

The content-addressed worker result has SHA-256
`fdd9859ccec9187facd47d6efa621449899d18b009a7b15ea746e45e1139ecb8`,
matching the remote manifest and settlement. The executed source matches the
attempt-D seal.

This is the exact requested state: Qwen2.5-7B-Instruct at revision
`a09a35458c702b33eeacc393d103063234e8bc28`, vLLM 0.23.0, PyTorch 2.11.0,
bf16, seed 41, a 16,384-token shared prefix, eight branches, 256 private suffix
tokens per branch, 1,024 shared plus 128 private blocks, and 1,056,964,608
logical bytes on a physical A100 80GB PCIe.

The correctness result is substantive, not an HBM proxy. All 1,152 source
allocator epochs are unique; all 1,152 post-free records have empty owners,
zero refcount, allocator availability, tombstoned epoch, inaccessible old
sessions, and no live branch reference. Assigned KV bytes fall from the exact
logical state size to zero. The destination contains 1,152 fresh committed
allocation lifetimes with zero physical-ID and zero allocation-lifetime overlap.
There are 1,152 page hashes, whole-state coverage, native raw-bit validation,
intact production engine-step binding, at least eight continuation tokens per
branch, and 8/8 first resumed tokens equal to independent recomputation.

Reducing all 10,454 canonical records independently yields 22,197,063,040
physical bytes, or **21.000762818351625x**. External movement is
2,114,200,960 bytes (2.000257098485553x); avoidable work is 12,684,381,568
bytes (12.000762818351625x). The dependency graph is closed and acyclic, the
10,454 IDs are unique, diagnostic bytes are zero, and record sums agree with
the published summary. Against frozen v10, this removes 39,106,884,224 bytes,
or 63.79178824422134%, and has 3.9992371816483754x of margin below the explicit
25x admission threshold. That is a genuine `MEANINGFUL` gate pass.

Attempts A-C do not create outcome-selection bias sufficient to reject attempt
D. A failed before model load. B and C failed closed before producing an
admissible restore outcome, respectively at stale source queue state and the
deterministic pending-event bound. The fixes were sealed, cause-aligned runtime
and accounting repairs; none changed workload, chunking, oracle, or go/no-go
threshold. This review does not authorize another micro repetition.

## Required qualifications

The micro timing is one instrumented observation on A100 PCIe, whereas frozen
v10 used A100-SXM4-80GB. Its byte/amplification result is admissible, but its
2.613090870-second export and 4.438193614-second restore must not be treated as
hardware-matched latency improvements over v10. The integrated run must use two
matching A100-80GB GPUs, and direct v10 timing comparison requires the frozen
SXM4 variant.

The reported 3.505231979 seconds of integrity work is specifically SHA-only
aggregate record time. It excludes raw-bit validation: 0.442472927 seconds of
CUDA events, 0.485534818 seconds of launch-wall intervals, and 0.502564720
seconds of synchronization attribution. Those views and the total 0.521796759
seconds of synchronization are not disjoint stages. They must not be added to
wall or CUDA time. The 10,454-event StatePassRecord mechanism is minimal
required instrumentation, but no disabled-instrumentation latency trial exists;
measured walls include its cost. Physical bytes mean canonical semantic memory
touches joined to actual operations and CUDA events, not raw memory-controller
transactions.

Temporary memory needs one correction in downstream reporting. The measured
58,720,256 bytes exactly matches the projected **pinned two-slot staging**
bound, but is not total host temporary memory. Restore RSS grows by 143,552,512
bytes, decomposed as 58,720,256 pinned plus 84,832,256 pageable. The conservative
v11 total host-temporary peak is therefore 143,552,512 bytes excluding the
resident 1,056,964,608-byte checkpoint. Against v10's 2,965,372,928 bytes, that
is a 20.65706051873199x reduction and 2,821,820,416 bytes saved.

Budget and cleanup pass. The ledger has 13,761.513626085978 consumed and
7,838.486373914022 remaining A100-seconds under the configured 21,600-second
ceiling. One 588-second two-GPU transaction needs 1,176 seconds, or 1,352.4
with the 15% planning margin. Attempt-D cleanup leaves no app, task, container,
endpoint, reservation, child process, profiler, or compute process; remote HBM
returns to 4 MiB.

## Mandatory integrated conditions

- Agent 10 must independently pass before reservation, and all four post-micro
  reviews must be sealed into one immutable preflight manifest.
- Run exactly one two-GPU integrated transaction with automatic retries
  disabled. Match the frozen v10 A100-SXM4-80GB hardware for direct timing
  comparison; fail scientific comparability on a different SKU.
- Keep the exact seed, model/runtime revisions, bf16 state workload, 12/15/20
  rps calibration, bounded queue trigger, and fail-closed scientific AND gate.
  Abort if the 12-rps or 15-rps guards materially contradict calibration.
- Preserve raw destination lifetime/mapping evidence and a public
  `IMPORT_ADMISSION` witness in the integrated artifact, in addition to epochs,
  all post-free records, StatePassRecords, and 8/8 oracle evidence.
- Use non-overlapping stage intervals for reclamation, restore, Amdahl, and
  economics. Keep GPU0 actively serving during restore and settle budget plus
  full cleanup after the invocation.
- Do not use attempt-D micro latency to claim v10 latency improvement, custom-
  hardware interest, or an end-to-end classification. Those require the valid
  integrated transaction.

Agents 9, 10, and 11 independently pass movement accounting, CUDA performance,
and Continuum correctness. All four post-micro reviews must now be sealed into
the immutable integrated preflight manifest.

Final adversarial decision: **PASS**. The qualifications above reduce the
claims, not the admission result. Exact correctness plus the measured fall from
58x to 21.000762818351625x warrants exactly one scientifically matched
integrated v11 transaction. This review does not authorize kill/recompute yet,
Experiment 005, FPGA work, or a hardware-interest conclusion.
