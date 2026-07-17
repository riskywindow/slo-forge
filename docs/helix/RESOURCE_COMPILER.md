# Learning-aware resource compiler

`compile_resource_plan` creates a deterministic per-tick CPU reference schedule for serving and
learning work. It models serving, rollout, environment, reward, verifier, training, and evaluation
resource vectors; mandatory serving demand; learning-value predictions; budgets; policy staleness;
privacy/effect controls; deadlines; preemption; and preservation.

Dedicated, static, utilization, FIFO, and Helix value-aware policies provide comparison baselines.
Traffic, GPU, CPU, storage, network, and value-prediction faults are explicit. Every tick records
capacity and class allocations, costs, active faults, serving predictions, work IDs, and audit
decisions. Serving SLO and capacity infeasibility raise errors before learning is admitted.

Predicted latency, queue depth, and learning value are not measurements. Scenario evidence references
must remain linked to raw inputs. See ADR 0046 and [capacity lending](CAPACITY_LENDING.md).
