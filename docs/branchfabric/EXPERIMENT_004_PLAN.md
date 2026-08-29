# Experiment 004 Plan: Real 2-GPU Capacity Reclamation + Cross-Layout State Movement

## Scope

Run two NVIDIA A100 80GB GPUs in one bounded Modal container. GPU 0 hosts a production-like serving pool; GPU 1 hosts long-running branch rollouts. No FPGA work is in scope.

## Sequence

1. Establish steady serving traffic on one logical pool and long-running branches on the other.
2. Inject a deterministic serving-load spike.
3. Pause rollout generation only at a validated safe boundary.
4. Capture a full checkpoint plus an explicitly versioned delta.
5. Reclaim GPU capacity and verify allocator/HBM release.
6. Perform the state transform and transfer across the source and destination layouts.
7. Validate block ownership, token history, model identity, and checkpoint/delta integrity.
8. Resume rollouts and compare outputs against an unmigrated deterministic control.

## Measurements

Measure pause latency, checkpoint and delta creation, capacity-reclamation latency, transform CPU/GPU time, transfer time and bandwidth, destination allocation, restore validation, resume latency, migration critical path, serving interference, HBM high-water marks, and final cleanup. Preserve raw timestamps and separate queueing, transfer, transformation, and GPU execution.

## Safety and budget

Use one explicitly authorized Modal function with `gpu="A100-80GB:2"`, a bounded GPU-hour ledger, no concurrent GPU coordinators, fresh-child ownership where applicable, and driver-visible cleanup gates for both UUIDs. Do not execute this plan as part of Experiment 003.
