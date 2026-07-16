# ADR 0013: Versioned canonical JSON at the Rust/Python boundary

- Status: Accepted
- Date: 2026-08-02

## Context

Python owns orchestration while Rust owns shared wire/protocol execution; language-local serialization could diverge.

## Decision

Exchange strict versioned JSON over bounded subprocess stdin/stdout and require golden round trips plus canonical hash agreement within an IR major version.

## Consequences

The boundary is inspectable and backward compatibility is testable. Binary payloads remain external content-addressed chunks rather than bloating JSON.
