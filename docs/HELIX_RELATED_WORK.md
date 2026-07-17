# Helix related work and claim boundary

Helix combines mechanisms from reinforcement learning, language-agent evaluation, replay systems,
inference serving, checkpointing, safe deployment, and provenance. It does not claim that policy
optimization, counterfactual branching, content addressing, canaries, or rollback are individually
novel. Its implemented contribution is systems composition: a deterministic local reference loop in
which capture, state reuse, policy provenance, reward, credit, training, resource admission, and
promotion carry strict evidence and fail closed at their boundaries.

## Online and preference optimization

PPO popularized clipped on-policy optimization [1], while DPO derives a direct preference objective
without the same online RL loop [2]. InstructGPT is a prominent large-scale RLHF pipeline combining
demonstrations, preference modeling, and PPO [3]. Helix's tiny reference trainer exposes policy-gradient,
clipped-ratio, KL-regularized, and pairwise modes so the evidence path is executable. It neither
reimplements a production RLHF stack nor demonstrates superior convergence or sample efficiency.

Distributed experience systems such as Acme [4], Reverb [5], RLlib [6], and SEED RL [7] separate
actors, replay/storage, learners, and evaluators at scale. Helix focuses on a different boundary: an
experience is admitted only with policy epoch, state, reward, credit, staleness, tenant, and artifact
provenance. Its scheduler models bounded learning work beside serving, but the checked implementation
is local CPU code rather than a distributed actor-learner runtime.

## Language agents and interactive environments

ReAct interleaves language reasoning and environment actions [8]. WebArena provides reproducible,
functional web environments [9], and SWE-agent exposes an agent-computer interface for repository
tasks [10]. These works motivate long-running trajectories whose files, tools, side effects, and
policy versions matter. Helix does not introduce a new agent benchmark. Its environment capsule and
effect ledger instead make a conservative subset of that state explicit for branching and replay.

The local demo's repository task is synthetic. It is not evidence that Helix improves WebArena,
SWE-bench, or production coding-agent performance.

## Counterfactuals, replay, and credit

Experience replay decouples data collection and updates in reinforcement learning [11]. Prioritized
experience replay ranks samples by predicted training utility [12]. Hindsight Experience Replay
relabels goals to recover signal from failures [13]. Helix's experience selector likewise considers
failure, uncertainty, novelty, rarity, safety, and expected value per cost, but keeps baseline policies,
privacy/effect gates, capacity, exclusion reasons, prediction uncertainty, and artifact hashes visible.
No checked experiment establishes that its value-aware score outperforms prioritized replay.

Pearl's structural causal framework distinguishes counterfactual claims from observational association
[14]. Helix captures sibling branches and can minimize declared interventions, but branch-relative
reward is not automatically causal: intervention labels are caller supplied, and hidden or undeclared
state can confound siblings. Exact replay claims are limited to declared model/environment identity;
the mode named causal compares declared event semantics and parent topology, not causal identification.

## Stateful inference and serving

Orca established iteration-level scheduling for transformer serving [15], and vLLM/PagedAttention
made KV-cache management central [16]. DistServe separates prefill and decode around goodput and
latency constraints [17]. Helix does not replace these data planes. It uses Continuum for explicit
model-state capture/reuse and SLOForge Fabric for resource/topology contracts, then enforces serving
feasibility before learning work. Current scheduler values and forecasts are supplied evidence, not
production measurements.

## Checkpointing and lineage

Content-addressed storage and copy-on-write are established systems techniques. CRIU demonstrates
process checkpoint/restore on Linux [18], while ML systems routinely checkpoint models and optimizer
state. Continuum's contribution to Helix is a typed execution-state compatibility boundary rather
than a claim that every runtime can be cloned. Environment capsules similarly capture declared local
state, not a complete machine or arbitrary remote service.

## Safe deployment and provenance

Site Reliability Engineering practice emphasizes staged rollout, canaries, monitoring, and rollback
[19]. Helix makes those stages evidence gates around a hash-addressed promotion capsule and preserves
active-session policy identity. Local SQLite transactions demonstrate fail-closed pointer update and
rollback; they are not a multi-region deployment controller or consensus protocol.

W3C PROV supplies a general model for entities, activities, and agents [20]. Helix uses narrower typed
lineage and artifact hashes suited to its learning loop. SHA-256 detects change only when an expected
digest is pinned independently; current capsules are not public-key signed.

## Implemented and unclaimed

Implemented evidence covers strict schemas, coordinated local capture, isolated local branching,
effect rejection, bounded replay, deterministic reward execution, branch-relative credit,
provenance-complete batches, staleness dispositions, a tiny trainer, selective experience curation,
serving-hard resource plans, local promotion/rollback, security controls, multi-seed CPU evaluation,
and a machine-readable fault campaign.

Helix does not claim a new RL algorithm, causal identification, production sample-efficiency gains,
GPU or multi-node performance, arbitrary environment replay, universal runtime compatibility,
distributed linearizability, cryptographic issuer authentication, or safe autonomous production
learning. See [limitations](helix/LIMITATIONS.md).

## References

1. J. Schulman et al. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017.
2. R. Rafailov et al. [Direct Preference Optimization](https://arxiv.org/abs/2305.18290), 2023.
3. L. Ouyang et al. [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155), 2022.
4. M. Hoffman et al. [Acme: A Research Framework for Distributed Reinforcement Learning](https://arxiv.org/abs/2006.00979), 2020.
5. A. Cassirer et al. [Reverb: A Framework for Experience Replay](https://arxiv.org/abs/2102.04736), 2021.
6. E. Liang et al. [RLlib: Abstractions for Distributed Reinforcement Learning](https://www.usenix.org/conference/osdi18/presentation/liang), OSDI 2018.
7. L. Espeholt et al. [SEED RL](https://arxiv.org/abs/1910.06591), ICLR 2020.
8. S. Yao et al. [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629), ICLR 2023.
9. S. Zhou et al. [WebArena](https://arxiv.org/abs/2307.13854), 2023.
10. J. Yang et al. [SWE-agent](https://arxiv.org/abs/2405.15793), 2024.
11. L.-J. Lin. [Self-Improving Reactive Agents Based on Reinforcement Learning, Planning and Teaching](https://doi.org/10.1007/BF00992699), 1992.
12. T. Schaul et al. [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952), 2015.
13. M. Andrychowicz et al. [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495), NeurIPS 2017.
14. J. Pearl. [Causality](https://www.cambridge.org/core/books/causality/B0046844FAE10CBF274D4ACBDAEB5F5B), second edition, 2009.
15. G. Yu et al. [Orca](https://www.usenix.org/conference/osdi22/presentation/yu), OSDI 2022.
16. W. Kwon et al. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), SOSP 2023.
17. Y. Zhong et al. [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin), OSDI 2024.
18. [CRIU project documentation](https://criu.org/Main_Page).
19. B. Beyer et al., eds. [Site Reliability Engineering](https://sre.google/sre-book/table-of-contents/), 2016.
20. W3C. [PROV-O: The PROV Ontology](https://www.w3.org/TR/prov-o/), 2013.
