# Restricted policy DSL

Genesis policies use a small deterministic S-expression language. The language cannot perform I/O, load code, allocate unbounded collections, spawn processes, call the network, or execute loops. A policy is an expression over fully declared scalar inputs and produces one bounded scalar output.

## Language

Inputs and outputs have type `int`, `float`, or `bool` and inclusive lower/upper bounds. Expressions support literals, variables, unary `not` and `neg`, arithmetic `add`, `sub`, `mul`, and `floor_div`, `min`, `max`, boolean `and`/`or`, comparisons, a conditional, and literal-bounded `clamp`. Programs declare an operation limit from 1 through 4096.

The checker rejects unknown variables/operators, type mismatches, invalid bounds, a potential zero denominator, output ranges wider than the declaration, and expressions over the operation limit. The bytecode compiler produces a fixed instruction tuple and the interpreter checks exact input names, types, bounds, stack safety, instruction count, and output bounds.

Example:

```text
policy slack_batch
input queue_length int 0 16
input priority int 0 10
input overloaded bool false true
output int 0 8
limit 64
return (clamp (if (or overloaded (ge priority 7)) (min queue_length 2) (min queue_length 8)) 0 8)
```

## Synthesis and explanation

The canonical formatter round-trips parsed programs. `policy_graph` emits an interpretable expression graph. The simplifier performs local constant and identity reductions. Seeded mutation changes literals and admits only candidates accepted by the same checker.

Bounded equivalence exhaustively enumerates Boolean and integer input domains, subject to an explicit state cap, and returns the first differing assignment as a counterexample. It intentionally rejects floating-point exhaustive domains. Equality within an enumerated domain is Level 3 bounded evidence, not a proof outside that domain.

## Integration and status

The deterministic local synthesis path emits a deadline/cancellation batching policy, compiles it to bounded bytecode, and records policy artifacts with the candidate. Cancellation CEGIS supplies the independent protocol check. Unit and property tests cover parsing, formatting, range/type rejection, compilation, interpretation, simplification, mutation, deterministic execution and bounded equivalence.

The DSL provides the mechanics for admission, batching, scheduling, cache, migration and recovery decisions when their inputs/outputs are encoded as scalars. The current local fixture exercises a request/serving batching policy alongside a separate bounded state-layout transformation; it does not claim that every listed policy family is already connected to a production runtime or that generated Rust source is emitted. The trusted policy target implemented today is bytecode plus the checked interpreter.
