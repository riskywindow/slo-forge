# BranchFabric characterization trace schemas

This directory specifies the event-oriented `BranchWorkloadTrace v1` and
`StateOperationTrace v1` measurement boundaries. They are separate from, and do
not replace, the legacy Helix workload-summary document in `schemas/helix`.

Both streams use one JSON object per event. Writers emit canonical JSON (UTF-8,
sorted keys, no insignificant whitespace, finite numbers only). `content_hash`
is lowercase SHA-256 over that canonical object after replacing `content_hash`
with 64 zeroes. A recorder assigns `event_sequence` before filtering or bounded
buffer admission, so sequence gaps are observable loss/sampling evidence.

Collection levels are:

- `disabled`: no events are admitted; omitted events are counted as filtered.
- `minimal`: lifecycle, COW, transaction, checkpoint, and migration boundaries.
- `full`: all events, subject only to declared deterministic sampling and buffer
  overflow.

JSONL is the authoritative streaming format. Parquet is an optional bulk format
with canonical JSON payloads plus indexed event columns. Perfetto/Chrome JSON is
derived visualization data, not raw measurement evidence. The compact trace
manifest records sampling, accepted/filtered/dropped event accounting, hardware
and software provenance, raw-artifact hashes, and the trace corpus hash.

The provenance discriminator is mandatory on every event and is one of
`SYNTHETIC`, `REPLAYED`, `HARDWARE_BACKED_REAL`, or `SIMULATED_HARDWARE`.
It describes workload evidence, not how the timer was read. The independent
`timing_measurement_class` field prevents real host timings for a synthetic
workload from falsely reclassifying that workload as hardware-backed real.

`branch_group_id` is explicitly nullable because capture can precede branch
group assignment; producers must not fabricate an identifier. In
`StateOperationTrace`, zero alignment, page size, or chunk size means N/A for a
nonpaged operation. It never means a fabricated one-byte granularity.
