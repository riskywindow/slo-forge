# Helix limitations

The implementation demonstrates a coherent, fail-closed local reference learning loop. Its claim
boundary is deliberately narrower than a production self-improving serving platform.

- Evaluation uses synthetic local CPU tasks and supplied scheduler predictions, not production traffic.
- The reference policy and trainer are tiny; no distributed LLM optimization or convergence claim is made.
- No GPU, multi-GPU, multi-node, cloud, or hardware co-design result is present in checked evidence.
- Environment capsules do not capture arbitrary kernel, network, credential, or undeclared remote state.
- Exact replay is available only inside declared captured identity; semantic replay is not equivalence.
- Branch-relative reward differences do not establish causal identification in uncontrolled systems.
- Controlled/observational intervention labels are evidence classifications supplied by the caller.
- Causal replay compares declared semantic events and parent topology; it does not prove causation.
- Trace comparison does not independently attest that model/environment bytes were restored.
- Learning-value and experience-selection features are predictions, not measured downstream gains.
- The resource compiler is discrete-tick and forecast-based; it does not guarantee real pause latency.
- SQLite coordinators and registries are local transactional components, not distributed consensus.
- SHA-256 capsules are not signed and authenticate nothing without independently pinned digests.
- Sandbox and isolation strength depend on host capabilities; callbacks cannot make external systems safe.
- Cross-tenant reuse, production capture, external effects, and paid deployment are disabled by default.
- The flagship task, candidate edits, failure-seed search, training schedule, and threshold are
  pre-authored; held-out cases are dataflow-isolated but not secret from the benchmark author.
- The implemented evaluation reports some hypotheses as partial or not exercised; it does not fill gaps
  with fabricated data.

Future work requires controlled production-like workloads, independent evaluator review, GPU and
distributed trainer adapters, linearizable coordination, stronger isolation, signed provenance, and
hardware-aware measurement with raw samples.
