# ADR 0047: Require a closed capsule and independent gates before promotion

## Context

A trainer or reward worker should not be able to promote its own candidate from one favorable metric.

## Decision

Build a hash-addressed promotion capsule with parent/candidate lineage, artifacts, rollback material, and
compatibility binding. Require independent lineage, reward-integrity, quality, safety, serving, and
compatibility gates followed by shadow and canary before atomic routing.

## Consequences

Admission authority is separated from proposal authority and rejection leaves the champion unchanged.
The process costs time and evidence storage; local gates do not establish production statistical power.
