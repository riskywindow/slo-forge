# Model-state branching

Helix delegates model execution-state capture and reuse to Continuum. `create_branch_group` supports
exact same-policy copy-on-write, controlled RNG mutation, compatible cross-policy reuse, and explicit
recomputation. Branch identifiers, leases, seeds, policy epochs, and strategies are validated before
forking.

Every successful member carries a `StateReuseReport` partitioning state into directly reused,
recomputed, replaced, and unsupported components with verification obligations. Unsupported state,
incompatible reuse, missing recomputation evidence, or strategy contradictions fail before a branch
is exposed. The report names the complete source-component universe, and its dispositions must cover
that universe exactly without overlaps or omissions. RNG mutation records the source and branch
counter state and cannot masquerade as exact transcript reuse.

Compatibility is scoped to Continuum's declared model/runtime contracts. It is not proof for an
unknown runtime or changed hidden state. See [Continuum compatibility](../continuum/COMPATIBILITY.md).
