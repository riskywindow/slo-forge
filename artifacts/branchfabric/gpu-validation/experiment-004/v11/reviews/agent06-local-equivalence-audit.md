# Agent 6 — v11 local equivalence audit

Result: **PASS (offline CPU equivalence)**

This independent review used the requested `PYTHONPATH=python:.venv/lib/python3.12/site-packages` with system PyTorch 2.10.0 and no CUDA device. It made no source or v10 changes and performed no Modal, GPU, or network work.

The v11 focused suite passed twice: 19/19 in 0.63 seconds and 19/19 in 0.64 seconds. Eighteen supplementary v11 runtime/methodology tests also passed. The randomized equivalence matrix covered seeds 41, 73, and 113 with one, two, and four layers, varied native block axes, shared/private pages, and partial tails. In every case, v11 export matched the frozen v10 canonical payload and manifest, and restored destination pages matched the v10 restore while untouched blocks remained unchanged.

Raw BF16 validation is genuinely bitwise: the test includes NaN payload, sign-bit, zero, and normal-value patterns, compares the restored `uint8` representation, and the implementation validates `int16` views rather than floating-point equality. Partial pages prove exact zero tails; full pages are directly overwritten without a preliminary full-page zero and remain equivalent to the frozen naive restore.

Integrity is layered and fail-closed. Whole-state and per-page SHA-256 commitments are verified, the manifest commitment binds the verified token, and each bounded snapshot is authenticated at consumption. Negative tests reject missing, duplicated, reordered, corrupted, post-verification-mutated, and NumPy-alias-mutated payloads before admission. Destination evidence binds ordered mapping, backing target, manifest, payload, and allocation epochs. Failure paths scrub selected native pages, validate raw zero, clean staged ownership, or escalate to typed runtime teardown.

For the exact 16K/fanout-8 topology, the deterministic planner produced 36 identical 28 MiB chunks, 32 pages per chunk, and 1,056,964,608 bytes total. Repeated descriptor hashes matched: `b5fcfc8ffa0aa24f0e859aeb9022d47ea99bd3b00be55845da1053ecf87d0c63`.

The frozen v10 adapter is unchanged from execution commit `c5fa2f169e8a343ef7a86ebc814b3a23a7de6b39`; both copies hash to `fa8f59b6e6755d0eb5c6c008d26df48b5fcbb4ad85f2af3b84233808c0ba953f`.

This PASS covers local CPU equivalence and negative semantics only. Real-vLLM continuation correctness and all CUDA measurements remain pending the authorized A100 micro-validation.
