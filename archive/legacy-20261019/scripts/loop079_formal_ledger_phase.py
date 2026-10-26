#!/usr/bin/env python3
"""Set a formal-ledger phase only at an externally verified request barrier.

The caller must establish that the prior phase has completed and that no prior
request is in flight.  A report hash is retained; this command does not infer
completion from timing or cohort ordinal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


def sha(data):
    return hashlib.sha256(data).hexdigest()


def set_phase(marker: Path, transitions: Path, run_id: str, phase: str,
              completed_report: Path | None):
    if phase == "warmup":
        if marker.exists():
            raise ValueError("warmup requires a fresh marker path")
        if completed_report is not None:
            raise ValueError("warmup has no preceding completed report")
        generation = 0
        previous = None
    else:
        if not marker.exists() or completed_report is None:
            raise ValueError("measured requires warmup marker and completed report")
        previous = json.loads(marker.read_text())
        if previous["run_id"] != run_id or previous["phase"] != "warmup" or previous["generation"] != 0:
            raise ValueError("preceding marker is not this run's warmup")
        generation = 1
    report_sha = sha(completed_report.read_bytes()) if completed_report else None
    row = dict(run_id=run_id, phase=phase, generation=generation,
               previous_marker_sha256=(sha(marker.read_bytes()) if previous else None),
               completed_client_report=str(completed_report) if completed_report else None,
               completed_client_report_sha256=report_sha,
               controller_monotonic_ns=time.monotonic_ns(),
               controller_pid=os.getpid())
    data = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
    marker.parent.mkdir(parents=True, exist_ok=True)
    transitions.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=marker.parent, delete=False) as temp:
        temp.write(data)
        temp.flush()
        os.fsync(temp.fileno())
        temp_path = Path(temp.name)
    try:
        os.replace(temp_path, marker)
    finally:
        temp_path.unlink(missing_ok=True)
    with transitions.open("ab") as out:
        out.write(data)
        out.flush()
        os.fsync(out.fileno())
    if marker.read_bytes() != data:
        raise ValueError("phase marker readback mismatch")
    return dict(**row, marker_sha256=sha(data))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=("warmup", "measured"))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--marker", type=Path, required=True)
    ap.add_argument("--transitions", type=Path, required=True)
    ap.add_argument("--completed-client-report", type=Path)
    a = ap.parse_args()
    result = set_phase(a.marker, a.transitions, a.run_id, a.phase,
                       a.completed_client_report)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
