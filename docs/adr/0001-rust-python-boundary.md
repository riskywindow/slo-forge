# ADR 0001: Versioned JSON is the primary Rust/Python boundary

Status: accepted

SLOForge uses JSON documents validated by the canonical DeploymentPlan schema for compiler inputs and outputs. Python invokes deterministic Rust tools as bounded subprocesses with input files or stdin and captures stdout, stderr, exit status, timeout, binary version, and artifact hashes. The running gateway uses HTTP with Server-Sent Events.

This boundary is easy to inspect, archive in an EvidenceBundle, replay across language versions, and exercise without a compiler toolchain at Python import time. It trades some serialization overhead outside the latency-sensitive request path for reproducibility and stable failure isolation. A future PyO3 accelerator may be added without becoming the canonical evidence format.
