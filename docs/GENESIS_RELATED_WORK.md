# Genesis related work and claim boundary

SLOForge Genesis combines ideas from inference serving, tensor compilation, bounded verification,
proof-carrying code, and safe deployment control. This document positions the implementation; it
does not claim that any individual mechanism is globally novel. The demonstrated claim is narrower:
within its declared CPU fixture and bounded verification domains, Genesis can recover a typed serving
description, generate a conservative runtime, propose structurally different implementations, reject
an invalid proposal with an independently reproduced counterexample, and carry scoped evidence into
later admission gates.

Links below point to primary papers, author/project pages, or official documentation. They are not an
exhaustive literature survey.

## Whole-server synthesis and inference serving

Modern serving systems establish the mechanisms over which Genesis searches. Orca introduced
iteration-level scheduling for transformer serving [1]. Clockwork uses centralized, predictable
execution to meet inference objectives [2], while INFaaS selects model, hardware, and optimization
variants [3]. AlpaServe searches model-parallel placements for statistical multiplexing [4]. These
systems demonstrate that scheduling and placement affect end-to-end objectives; they do not make a
generated implementation trustworthy merely because it benchmarks well.

PagedAttention and vLLM made KV-cache management and continuous batching central serving concerns
[5]. SGLang couples a language for structured model programs with runtime mechanisms such as prefix
reuse [6]. DistServe separates prefill and decode to optimize goodput under TTFT and TPOT constraints
[7], and Sarathi-Serve uses chunked prefills to control interference [8]. Genesis represents these
classes of decisions in `ServingGenome`, `StateGenome`, `WorkflowGenome`, and `DistributedGenome`.
Its current CPU runtime is intentionally more conservative: it serializes reference operations inside
bounded microbatches and does not claim the cluster-scale throughput of these systems.

SLOForge Genesis differs from a configuration wrapper in two ways. First, its transformation IR can
change policy, tensor, state, distributed, kernel, recovery, and generated runtime structure, subject
to explicit obligations. Second, candidate authority belongs to independent verification, capsule
validation, and promotion control, not to the proposal engine. The present artifact does not establish
that Genesis explores all those surfaces jointly or outperforms the systems above on GPUs.

The current zero-day AST path also draws a conservative compiler boundary: it recovers a typed call
and state inventory but not SSA value bindings or output tensor metadata. The `TensorGenome` therefore
persists an `unresolved_static_call_inventory` and enables no algebraic rewrite over those calls.
Genesis does not count a syntax-level call list as the executable tensor graph that systems such as
TVM, TASO, or `torch.export` operate on.

## Tensor compilers, rewrite systems, and kernel search

Halide separates an image-processing algorithm from its schedule [9]. TVM applies a similar compiler
and optimization framing to tensor programs [10], and Ansor searches automatically generated tensor
programs using a learned cost model [11]. TASO applies verified algebraic graph substitutions to deep
learning computation graphs [12]. FlexFlow searches parallelization strategies for DNN graphs [13].
Genesis inherits the central separation between semantics, transformations, schedules, and measured
cost, while extending the object under search to persistent state and serving protocols.

Equality saturation, as implemented by `egg`, provides a principled way to defer rewrite ordering and
extract a low-cost equivalent expression [14]. Genesis currently implements a smaller, explicit,
bounded tensor rewrite explorer rather than a general e-graph. Every rule declares shapes, dtypes,
numerical mode, and verifier obligations; floating-point reassociation is not silently classified as
exact.

Triton shows how a restricted language and compiler can make focused GPU kernels programmable and
searchable [15]. Genesis's kernel lab follows that evidence discipline for a narrow quantized-state
update surface: candidates have typed domains, a reference path, randomized/edge-case checks, and raw
timing samples. The checked-in evaluation environment had no NVIDIA GPU or CUDA compiler, so Triton
execution and GPU speedups remain unexercised. Genesis does not attempt to replace general attention
kernel libraries.

## Counterexample-guided synthesis and independent checking

Sketching and counterexample-guided inductive synthesis establish the propose/check/counterexample
loop used by Genesis [16]. The key architectural lesson is that a proposal is refined using witnesses
from a separate checker. Genesis applies this pattern to typed serving candidates: it persists the
original failure, minimizes a witness by replay, learns a typed precondition, suppresses repeated
family failures, and rechecks the corrected candidate.

QuickCheck established practical randomized property testing from executable specifications [17].
Genesis combines seeded differential cases, schema-aware adversarial generation, metamorphic or
property checks, and bounded explicit-state protocol exploration. These are complementary evidence
sources with independent claim levels, not a single `verified` Boolean.

The current CEGIS demonstration is deliberately bounded. It finds a cancellation/emit schedule,
reduces it from `admit, schedule, prefill, decode, cancel, emit` to `admit, cancel, emit`, and learns
that the batching-policy family must check cancellation immediately before emission. This is a real
fixture execution, but it is not a proof for arbitrary programs, unbounded schedules, or external
runtimes.

## Proof-carrying artifacts and compiler correctness

Proof-carrying code separates an untrusted code producer from a consumer that checks evidence against
a safety policy [18]. CompCert demonstrates the value of a small, explicit correctness argument for a
compiler rather than trusting optimization testing alone [19]. Genesis adopts the producer/checker
separation but makes a more modest claim: a `GenesisCapsule` is hash-addressed evidence for scoped
claims, not a universal machine-code proof and not a public-key signature.

Capsule validation independently checks canonical hashes, artifact integrity, contract and hardware
scope, dependency versions, evidence freshness, raw performance provenance, rollback material, and
promotion completeness. The validator does not trust issuer labels found inside the capsule: an
operator-supplied context outside the capsule pins the expected capsule digest and trusted evidence
anchors, including issuer/version, evidence-record digest, and exact artifact digests. A claim records
its own verification level from build evidence through hardware/operational evidence. Generated
source, proposal scores, and synthesis-agent reasoning remain outside the trusted computing base.

## Runtime verification and bounded model checking

TLA+ and its model checker popularized executable specifications and finite-state exploration for
concurrent/distributed protocols [20]. SPIN likewise explores communicating-process models and emits
counterexample executions [21]. Genesis uses a self-contained Rust explicit-state checker so normal CI
does not depend on an external formal-methods installation. Its model covers admission, queues,
streaming token commitment, cancellation, retry, state ownership/transfer, worker failure,
champion-challenger rollout, controller crash/restart, promotion, and rollback.

Every result records bounds, assumptions, state and transition counts, individual invariants, and a
replayable trace. Truncated search is inconclusive and `universal_proof` is always false. The checker
does not model tensor arithmetic, packet networks, CUDA/NCCL internals, weak memory, or unbounded
liveness.

## Distributed inference and communication planning

Megatron-LM demonstrated intra-layer model parallelism for transformers [22]. GSPMD expresses broad
parallel strategies through sharding annotations [23], Alpa automates inter- and intra-operator
parallelism [24], and FlexFlow searches device placement/parallelization [13]. DistServe [7] further
shows that serving phase separation and transfer costs must be included in distributed planning.

Genesis reuses SLOForge Fabric's topology graph, physical-plan validators, measured fabric curves,
placement, and recovery variants. Its distributed synthesis surface mutates selected fields of an
already valid physical plan and invalidates stale performance evidence. It does not currently
synthesize arbitrary parallel degrees or claim a real multi-node Genesis result.

## Workflow-aware serving

SGLang's structured programs make control flow, prefix reuse, and multi-call execution visible to the
runtime [6]. Genesis's `WorkflowGenome` similarly represents model calls, tools, verification passes,
branches, loops, deadlines, cancellation, shared prefixes, and expected future requests. The purpose
is different: the workflow becomes an input to cross-layer synthesis and proof obligations, not just a
runtime programming interface. The current CPU flagship contains a declared optional workflow; live
tool execution and external workflow traffic have not been evaluated.

## Self-healing systems and safe evolution

Borg demonstrates large-scale declarative deployment, health management, and rescheduling [25].
Operational practice also motivates staged rollout, canary analysis, and rollback [26]. Genesis builds
on SLOForge's existing canary, rollback, Autopsy, and recovery machinery, adding content-addressed
champion/challenger identities and a capsule gate.

The flagship local flow builds two real capsules from separately synthesized candidates, validates
both against their respective trust contexts, registers the second as an isolated challenger,
persists every transition, and revalidates the challenger immediately before promotion. Active streams
remain leased to the capsule on which they began and the prior champion remains available for
rollback. Its shadow/canary observations are supplied deterministic local evidence. The subsequent
fabric-degradation trigger is modeled; none of this is evidence of external production traffic,
physical state transfer, or autonomous mutation of a live deployment.

## What Genesis implements and does not claim

The implementation supports:

- a versioned, typed eight-region serving genome and transformation/candidate/counterexample IRs;
- conservative zero-day AST inspection with unsupported behavior made explicit;
- deterministic generation of a bounded reference runtime;
- restricted policy, tensor, state, distributed, and focused kernel synthesis surfaces;
- budgeted search and scoped independent operator, quality, resource, statistical, and protocol
  checks;
- counterexample minimization, learned family constraints, lineage storage/transfer/invalidation;
- hash-addressed capsules, fail-closed local sandboxing, and persisted local evolution control; and
- a deterministic CPU ServingSynthBench profile with hidden evaluator cases.

It does not claim:

- globally novel batching, scheduling, tensor, kernel, placement, or rollout algorithms;
- universal Python/model support, semantic equivalence, memory safety, or protocol correctness;
- global search optimality or general performance improvement;
- public-key-authenticated capsules;
- GPU, multi-GPU, multi-node, cloud, or production-traffic validation in the checked-in CPU evidence;
- Linux sandbox or Docker-daemon execution on the checked Darwin host; or
- that representing a transformation implies that a lowering and verifier exist for every possible
  combination.

## References

1. G. Yu et al. [Orca: A Distributed Serving System for Transformer-Based Generative
   Models](https://www.usenix.org/conference/osdi22/presentation/yu). OSDI 2022.
2. A. Gujarati et al. [Serving DNNs like Clockwork: Performance Predictability from the Bottom
   Up](https://www.usenix.org/conference/osdi20/presentation/gujarati). OSDI 2020.
3. F. Romero et al. [INFaaS: Automated Model-less Inference
   Serving](https://www.usenix.org/conference/atc21/presentation/romero). USENIX ATC 2021.
4. Z. Li et al. [AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning
   Serving](https://arxiv.org/abs/2302.11665). OSDI 2023.
5. W. Kwon et al. [Efficient Memory Management for Large Language Model Serving with
   PagedAttention](https://arxiv.org/abs/2309.06180). SOSP 2023.
6. L. Zheng et al. [SGLang: Efficient Execution of Structured Language Model
   Programs](https://arxiv.org/abs/2312.07104). 2024.
7. Y. Zhong et al. [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large
   Language Model Serving](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin).
   OSDI 2024.
8. A. Agrawal et al. [Taming Throughput-Latency Tradeoff in LLM Inference with
   Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal). OSDI 2024.
9. J. Ragan-Kelley et al. [Halide: A Language and Compiler for Optimizing Parallelism, Locality,
   and Recomputation in Image Processing Pipelines](https://doi.org/10.1145/2491956.2462176).
   PLDI 2013.
10. T. Chen et al. [TVM: An Automated End-to-End Optimizing Compiler for Deep
    Learning](https://www.usenix.org/conference/osdi18/presentation/chen). OSDI 2018.
11. L. Zheng et al. [Ansor: Generating High-Performance Tensor Programs for Deep Learning](https://www.usenix.org/conference/osdi20/presentation/zheng).
    OSDI 2020.
12. Z. Jia et al. [TASO: Optimizing Deep Learning Computation with Automatic Generation of Graph
    Substitutions](https://doi.org/10.1145/3341301.3359630). SOSP 2019.
13. Z. Jia et al. [Beyond Data and Model Parallelism for Deep Neural Networks](https://arxiv.org/abs/1807.05358).
    MLSys 2019.
14. M. Willsey et al. [egg: Fast and Extensible Equality Saturation](https://doi.org/10.1145/3434304).
    POPL 2021.
15. P. Tillet et al. [Triton: An Intermediate Language and Compiler for Tiled Neural Network
    Computations](https://doi.org/10.1145/3315508.3329973). MAPL 2019.
16. A. Solar-Lezama et al. [Combinatorial Sketching for Finite
    Programs](https://doi.org/10.1145/1168857.1168907). ASPLOS 2006.
17. K. Claessen and J. Hughes. [QuickCheck: A Lightweight Tool for Random Testing of Haskell
    Programs](https://doi.org/10.1145/351240.351266). ICFP 2000.
18. G. C. Necula. [Proof-Carrying Code](https://doi.org/10.1145/263699.263712). POPL 1997.
19. X. Leroy. [Formal Verification of a Realistic Compiler](https://doi.org/10.1145/1538788.1538814).
    Communications of the ACM, 2009.
20. L. Lamport. [Specifying Systems: The TLA+ Language and Tools for Hardware and Software
    Engineers](https://lamport.azurewebsites.net/tla/book.html). Addison-Wesley, 2002.
21. G. J. Holzmann. [The SPIN Model Checker](https://spinroot.com/spin/whatispin.html). Official
    project overview and references.
22. M. Shoeybi et al. [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model
    Parallelism](https://arxiv.org/abs/1909.08053). 2019.
23. Y. Xu et al. [GSPMD: General and Scalable Parallelization for ML Computation
    Graphs](https://arxiv.org/abs/2105.04663). 2021.
24. L. Zheng et al. [Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep
    Learning](https://www.usenix.org/conference/osdi22/presentation/zheng-lianmin). OSDI 2022.
25. A. Verma et al. [Large-scale Cluster Management at Google with
    Borg](https://doi.org/10.1145/2741948.2741964). EuroSys 2015.
26. B. Beyer et al., eds. [Site Reliability Engineering: How Google Runs Production
    Systems](https://sre.google/sre-book/table-of-contents/). O'Reilly, 2016.
