from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from sloforge.trace.format import TraceRequest, generate_bursty_trace, validate_trace


@given(
    arrivals=st.lists(
        st.floats(min_value=0, max_value=1_000_000, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=80,
    )
)
@settings(max_examples=50, deadline=None)
def test_sorted_unique_traces_validate(arrivals: list[float]) -> None:
    requests = [
        TraceRequest(
            request_id=f"r-{index}",
            arrival_ms=arrival,
            prompt_tokens=1 + index,
            output_tokens=1,
        )
        for index, arrival in enumerate(sorted(arrivals))
    ]
    summary = validate_trace(requests)
    assert summary.request_count == len(arrivals)
    assert summary.peak_one_second_rps >= 1


def test_generator_has_bursts_priorities_and_long_contexts() -> None:
    first = generate_bursty_trace(seed=19, count=90)
    second = generate_bursty_trace(seed=19, count=90)
    assert first == second
    summary = validate_trace(first)
    assert len(summary.priorities) >= 2
    assert {item.request_class for item in first} == {"interactive", "long-context"}
    assert summary.peak_one_second_rps > summary.mean_arrival_rate_rps


def test_duplicate_and_unsorted_requests_fail_closed() -> None:
    base = TraceRequest(request_id="same", arrival_ms=10, prompt_tokens=2, output_tokens=2)
    duplicate = base.model_copy(update={"arrival_ms": 11})
    try:
        validate_trace([base, duplicate])
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate IDs must be rejected")
    later = base.model_copy(update={"request_id": "later", "arrival_ms": 9})
    try:
        validate_trace([base, later])
    except ValueError as exc:
        assert "nondecreasing" in str(exc)
    else:
        raise AssertionError("unsorted arrivals must be rejected")
