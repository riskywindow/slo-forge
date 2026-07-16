# State-conversion compiler

The compiler maps a compatible source/destination layout pair to a typed DAG, chunk schedule, bounded memory plan, and verification plan. It rejects logical-shape mismatches, missing shards, invalid page coverage, insufficient memory, and an exactness requirement that the requested operations cannot satisfy.

## Backends

- Canonical CPU: decode source physical state into canonical logical K/V and encode the destination. This is the trusted correctness path.
- Direct CPU: read only a token chunk from source shards and immediately write destination shards, avoiding full canonical materialization.
- Streaming direct CPU: yields destination chunks under the schedule's maximum in-flight memory bound.
- Quality-bounded CPU dtype conversion: executes floating-point narrowing only with an explicit maximum-absolute-error budget, compares all destination bytes with the canonical converter, measures source-to-destination loss, and fails when the observed loss exceeds the budget.
- PyTorch explicit-device converter: performs the same bounded layout transformation on exactly the requested CPU or CUDA device and returns only after complete comparison with the independent canonical converter. It never silently falls back between devices.

No GPU conversion result is claimed on the current Apple Silicon CPU environment. The CUDA path and its independently verified GPU test are implemented but unexercised here; no custom Triton/CUDA kernel is present. Any future generated GPU kernel must match the trusted converter under randomized layout/stride/page/quantization tests before selection.

## Direct conversion exercised

`direct_convert_capture` converts the real reference runtime's int32 attention segments from token-major/separate K/V/TP=4/page=3 to head-major/packed K/V/TP=2/page=5. It computes destination segments in bounded scratch space and compares their bytes with an independently encoded canonical result. The migration artifact records this evidence; recurrent and control state are handled as separately typed logical components.

## Evidence-based selection

`measure_and_select_converter` measures repeated canonical and direct executions on the local host, verifies exact output equality, and selects from those observed samples. Timing is host-observed and retains raw provenance; simulated transport time remains a separate metric class. A benchmark result must never be interpreted as a GPU or production-network measurement.
