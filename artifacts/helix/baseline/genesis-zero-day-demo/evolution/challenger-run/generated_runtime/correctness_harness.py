"""Differential fixture harness for this generated reference runtime."""

import argparse
import json
from pathlib import Path

from runtime import application


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    args = parser.parse_args()
    app = application(seed=args.seed)
    failures = []
    cases = []
    try:
        app.start()
        for line_number, line in enumerate(args.samples.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            handle = app.runtime.submit_text(
                request_id=f"fixture-{line_number}",
                text=sample["text"],
                maximum_new_tokens=sample["maximum_new_tokens"],
                seed=sample["seed"],
                timeout_seconds=args.timeout_seconds,
            )
            observed = [event.token_id for event in handle.events(args.timeout_seconds) if event.token_id is not None]
            expected = sample.get("expected_tokens")
            exact = expected is None or observed == expected
            cases.append(
                {
                    "line": line_number,
                    "request_seed": sample["seed"],
                    "expected": expected,
                    "observed": observed,
                    "exact_match": exact,
                }
            )
            if expected is not None and observed != expected:
                failures.append({"line": line_number, "expected": expected, "observed": observed})
    finally:
        app.shutdown()
    print(json.dumps({"cases": cases, "failures": failures, "passed": bool(cases) and not failures}, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
