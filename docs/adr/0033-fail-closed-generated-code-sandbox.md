# ADR 0033: Execute generated code only through a fail-closed OS sandbox

- Status: accepted
- Date: 2026-08-02

## Context

Generated programs may exfiltrate data, discover credentials, alter verifier inputs, fork
unbounded children, flood output, consume resources, or persist outside their artifact directory.
Language-level filtering alone is insufficient for general generated runtimes.

## Decision

Invoke argv directly without a shell, rebuild the environment from an allowlist, close stdin, use
explicit read-only inputs and one separate writable artifact directory, enforce bounded time/
resources/output, validate the output tree, and kill the process group on failure. Require kernel
network and filesystem isolation for strict execution. Use macOS `sandbox-exec` on Darwin and
bubblewrap namespaces on Linux. If the capability is missing or setup fails, return a typed failure
without running the generated program. Do not provide GPU/device access by default.

The macOS backend has been exercised locally; it is profile-based, deprecated by Apple, does not
create a device namespace, and does not provide a system-wide file-read allowlist. The Linux
bubblewrap adapter is implemented but was not exercised in this workspace. Memory and Linux
process-count limits remain best effort without cgroups. Windows strict execution is unavailable.

## Consequences

Normal local synthesis cannot silently weaken isolation or inherit credentials. Generated output is
bounded and separated from verifier and baseline files. Kernel/runtime sandbox flaws and readable
system paths remain residual risks, so the sandbox is one layer rather than proof that code is safe.
GPU execution and alternative platforms need separately tested backends and explicit opt-in.
