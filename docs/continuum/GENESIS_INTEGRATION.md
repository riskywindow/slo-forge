# Genesis integration

Genesis describes generated-runtime state through its runtime descriptor and generated configuration. The Continuum binding validates that configuration, loads the generated baseline runtime, and exercises a deterministic CPU smoke path where available.

A deeper generated adapter must emit logical component semantics, physical layout, dependency edges, compatible transformations, recomputation hooks, and proof obligations. Continuum may consume a Genesis-generated pack/import/recurrent conversion, but generated code remains untrusted until the trusted canonical converter and continuation verifier accept it.

GenesisCapsules and `ExecutionStateCapsule` remain separate artifacts linked by digests. The former records synthesis lineage and runtime evidence; the latter records portable session state, transaction binding, and ownership. Neither opaque artifact subsumes the other.

The present adapter is a versioned binding and CPU smoke exercise, not evidence of a hardware-backed live migration into a Genesis-generated runtime.
