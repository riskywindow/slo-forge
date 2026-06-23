# ServingSynthBench baselines

The schema names all required comparison surfaces. CPU smoke evaluates only surfaces that can execute the dependency-free task honestly.

| Baseline | CPU smoke behavior |
| --- | --- |
| Python eager reference | FIFO reference execution |
| PyTorch eager reference | Not applicable: the smoke grammar is not a `torch.nn.Module` |
| `torch.compile` | Not applicable: no Torch graph exists in this profile |
| Generic runtime adapter | Conservative FIFO scheduling |
| Tuned static SLOForge | Deterministically selects among FIFO, shortest-prompt, and earliest-deadline orders |
| Physical-plan-only SLOForge | Not applicable without GPU fabric topology |
| Policy-only search | Selects between deadline and priority policies |
| Kernel-only search | Not applicable without a GPU kernel target |
| Genesis without Autopsy | Local FIFO/shortest ablation |
| Genesis without counterexample learning | Local FIFO/shortest ablation |
| Genesis without lineage | Local FIFO/shortest ablation |
| Genesis full | Selects among deadline, priority, shortest-prompt, and shared-prefix orders |
| Deterministic single-shot | One FIFO proposal |

Candidate selection uses a deterministic scheduling cost and stable tie breaking. All applicable baselines execute the same reference implementation and inputs; consequently the smoke benchmark primarily checks orchestration, evidence completeness, and semantic validity. The ablations are explicit local algorithms, not claims that the complete production subsystems were exercised. Hardware-only entries remain present with machine-readable reasons instead of fabricated values.

