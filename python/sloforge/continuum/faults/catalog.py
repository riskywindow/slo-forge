"""Versioned Continuum fault catalog with fail-safe expected responses."""

from __future__ import annotations

from .models import FaultComponent, FaultDefinition, FaultKind


def _definition(
    kind: FaultKind,
    component: FaultComponent,
    phase: str,
    response: str,
) -> FaultDefinition:
    return FaultDefinition(
        kind=kind,
        ground_truth_label=f"continuum.fault.{kind.value}",
        affected_component=component,
        transaction_phase=phase,
        expected_protocol_response=response,
    )


FAULT_CATALOG: tuple[FaultDefinition, ...] = (
    _definition(
        FaultKind.SOURCE_CRASH_BEFORE_INITIAL_SNAPSHOT,
        FaultComponent.SOURCE_RUNTIME,
        "DESTINATION_PREPARING",
        "abort before commit; preserve the current source lease or classify source loss",
    ),
    _definition(
        FaultKind.SOURCE_CRASH_DURING_PRECOPY,
        FaultComponent.SOURCE_RUNTIME,
        "PRECOPYING",
        "abort before commit or recover the source dirty log without moving ownership",
    ),
    _definition(
        FaultKind.SOURCE_CRASH_AFTER_QUIESCE,
        FaultComponent.SOURCE_RUNTIME,
        "SOURCE_FROZEN",
        "finish only with validated final state; otherwise enter explicit source-lost recovery",
    ),
    _definition(
        FaultKind.SOURCE_CRASH_AFTER_COMMIT_INTENT,
        FaultComponent.SOURCE_RUNTIME,
        "COMMIT_INTENT_RECORDED",
        "recover the persisted intent and commit once or enter operator-required state",
    ),
    _definition(
        FaultKind.DESTINATION_CRASH_DURING_IMPORT,
        FaultComponent.DESTINATION_RUNTIME,
        "DESTINATION_IMPORTING",
        "abort destination and resume the current source owner",
    ),
    _definition(
        FaultKind.DESTINATION_CRASH_DURING_VALIDATION,
        FaultComponent.DESTINATION_RUNTIME,
        "DESTINATION_VALIDATING",
        "persist abort and rollback before commit; resume the source at the unchanged epoch",
    ),
    _definition(
        FaultKind.DESTINATION_CRASH_AFTER_OWNERSHIP_COMMIT,
        FaultComponent.DESTINATION_RUNTIME,
        "OWNERSHIP_COMMITTED",
        "do not roll back to the stale source; recover destination or begin a new migration",
    ),
    _definition(
        FaultKind.GATEWAY_CRASH_DURING_SWITCH,
        FaultComponent.GATEWAY,
        "GATEWAY_SWITCHING",
        "replay the idempotent owner switch from the committed epoch and watermark",
    ),
    _definition(
        FaultKind.COORDINATOR_CRASH,
        FaultComponent.COORDINATOR,
        "ANY",
        "recover the durable journal before allowing another ownership mutation",
    ),
    _definition(
        FaultKind.DELAYED_COORDINATOR_RESPONSE,
        FaultComponent.COORDINATOR,
        "ANY",
        "respect the transaction deadline and retry only idempotent compare-and-swap operations",
    ),
    _definition(
        FaultKind.STALE_SOURCE_WRITER,
        FaultComponent.SOURCE_RUNTIME,
        "DESTINATION_ACTIVE",
        "reject mutation and output carrying the stale owner epoch",
    ),
    _definition(
        FaultKind.STALE_DESTINATION_EPOCH,
        FaultComponent.DESTINATION_RUNTIME,
        "GATEWAY_SWITCHING",
        "reject activation because the prepared epoch differs from the committed epoch",
    ),
    _definition(
        FaultKind.NETWORK_PARTITION,
        FaultComponent.TRANSPORT,
        "PRECOPYING",
        "bound retries and abort or retain rollback state at deadline",
    ),
    _definition(
        FaultKind.TRANSFER_TIMEOUT,
        FaultComponent.TRANSPORT,
        "FINAL_DELTA_TRANSFERRING",
        "fail the bounded transfer and preserve the pre-commit source",
    ),
    _definition(
        FaultKind.CHUNK_LOSS,
        FaultComponent.TRANSPORT,
        "PRECOPYING",
        "retransmit by content hash within the configured retry bound",
    ),
    _definition(
        FaultKind.CHUNK_DUPLICATION,
        FaultComponent.TRANSPORT,
        "PRECOPYING",
        "deduplicate by authenticated content hash",
    ),
    _definition(
        FaultKind.CHUNK_CORRUPTION,
        FaultComponent.TRANSPORT,
        "PRECOPYING",
        "reject the chunk and retransmit or abort",
    ),
    _definition(
        FaultKind.CHECKSUM_MISMATCH,
        FaultComponent.TRANSPORT,
        "DESTINATION_VALIDATING",
        "reject destination validation and abort before commit",
    ),
    _definition(
        FaultKind.OUT_OF_ORDER_DELTA,
        FaultComponent.DIRTY_TRACKER,
        "DELTA_SYNCING",
        "reject a stale or gapped delta epoch",
    ),
    _definition(
        FaultKind.DIRTY_LOG_OVERFLOW,
        FaultComponent.DIRTY_TRACKER,
        "DELTA_SYNCING",
        "restart from a full snapshot or select stop-and-copy",
    ),
    _definition(
        FaultKind.HIGH_DIRTY_RATE,
        FaultComponent.DIRTY_TRACKER,
        "DELTA_SYNCING",
        "stop non-convergent rounds at the configured bound and select hybrid stop-and-copy",
    ),
    _definition(
        FaultKind.DESTINATION_OOM,
        FaultComponent.DESTINATION_RUNTIME,
        "DESTINATION_PREPARING",
        "abort destination allocation without moving ownership",
    ),
    _definition(
        FaultKind.CONVERSION_KERNEL_FAILURE,
        FaultComponent.CONVERTER,
        "DESTINATION_IMPORTING",
        "reject unverified output and use only an explicitly planned verified fallback",
    ),
    _definition(
        FaultKind.DESTINATION_INCOMPATIBILITY,
        FaultComponent.DESTINATION_RUNTIME,
        "COMPATIBILITY_VALIDATED",
        "reject migration before active state transfer",
    ),
    _definition(
        FaultKind.PAGE_TABLE_CORRUPTION,
        FaultComponent.DESTINATION_RUNTIME,
        "DESTINATION_VALIDATING",
        "fail structural validation and abort before commit",
    ),
    _definition(
        FaultKind.TOKEN_GAP,
        FaultComponent.GATEWAY,
        "ANY",
        "reject the token and retain the expected sequence index",
    ),
    _definition(
        FaultKind.DUPLICATE_TOKEN,
        FaultComponent.GATEWAY,
        "ANY",
        "deduplicate an identical event and reject a mismatched payload",
    ),
    _definition(
        FaultKind.CLIENT_DISCONNECT,
        FaultComponent.CLIENT,
        "GATEWAY_SWITCHING",
        "retain the gateway watermark for sequence-based resume",
    ),
    _definition(
        FaultKind.CANCELLATION_DURING_CUTOVER,
        FaultComponent.CLIENT,
        "SOURCE_QUIESCING",
        "serialize cancellation with cutover and commit terminal output once",
    ),
    _definition(
        FaultKind.CLOCK_SKEW,
        FaultComponent.COORDINATOR,
        "ANY",
        "use coordinator versions and monotonic deadlines instead of runtime wall clocks",
    ),
    _definition(
        FaultKind.WARM_DESTINATION_REGRESSION,
        FaultComponent.WARMPATH,
        "DESTINATION_PREPARING",
        "fail readiness validation and retain the source owner",
    ),
)

_BY_KIND = {definition.kind: definition for definition in FAULT_CATALOG}
if len(_BY_KIND) != len(FaultKind):
    raise RuntimeError("Continuum fault catalog must define every fault kind exactly once")


def fault_definition(kind: FaultKind) -> FaultDefinition:
    return _BY_KIND[kind]
