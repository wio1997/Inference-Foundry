"""Recover one 32K AISBench phase's client admission and request duration bounds."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from audit_SG_32K_W1_latency_timeline_g1375 import load_times, summarize


def audit(phase_dir, phase_audit):
    reference_bytes = phase_audit.read_bytes()
    reference = json.loads(reference_bytes)
    count = reference["requests"]
    if not reference["fully_validated"] or count not in {16, 32}:
        raise ValueError("Expected a fully audited 16- or 32-request phase")
    detail, = phase_dir.rglob("*_details.jsonl")
    raw = detail.read_bytes()
    raw_sha = sha256(raw).hexdigest()
    if reference["evidence"][str(detail.relative_to(phase_dir))] != raw_sha:
        raise ValueError("Raw request file differs from the completed phase")
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != count or {row["id"] for row in rows} != set(range(count)):
        raise ValueError("Missing or repeated request IDs")
    if {row["data_id"] for row in rows} != set(range(count)) or not all(
        row["success"] is True and not row["error_info"] and row["input_tokens"] == 32768
        and row["output_tokens"] == 1024 for row in rows
    ):
        raise ValueError("Raw request shape or status does not match the audited phase")
    clocks, db_pins = load_times(detail, rows)
    timeline = summarize(rows, clocks, reference)
    if phase_audit.read_bytes() != reference_bytes or detail.read_bytes() != raw:
        raise ValueError("An original artifact changed during the audit")

    starts = sorted(times[0] for times in clocks)
    first_completion = min(times[-1] for times in clocks)
    spans = timeline["per_request_first_token_order"]
    before_completion = sum(start < first_completion for start in starts)
    next_admission = (starts[16] - first_completion) if count == 32 else None
    return {
        "request_count": count, "input_tokens": 32768, "output_tokens": 1024,
        "phase_duration_s": timeline["recomputed_duration_s"],
        "peak_client_concurrency": timeline["peak_concurrency"],
        "new_requests_started_before_first_completion": before_completion,
        "request17_start_minus_first_completion_s": next_admission,
        "maximum_client_request_E2EL_s": max(row["E2EL_s"] for row in spans),
        "maximum_client_request_TTFT_s": max(row["TTFT_s"] for row in spans),
        "first_to_last_client_start_s": starts[-1] - starts[0],
        "raw_details_SHA256": raw_sha, "audit_SHA256": sha256(reference_bytes).hexdigest(),
        "timing_database_SHA256": db_pins,
        "scope": "One AISBench client clock. Start is recorded before HTTP submit, TTFT is first SSE data.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase_dir", type=Path)
    parser.add_argument("phase_audit", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.phase_dir.resolve(strict=True), args.phase_audit), indent=2, sort_keys=True))
