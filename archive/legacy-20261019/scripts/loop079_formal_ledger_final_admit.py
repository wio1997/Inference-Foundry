#!/usr/bin/env python3
"""Fail closed after Run533 stop, source restore, and two-phase reduction."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args()
    root = a.run_dir
    statuses = dict(line.split("=", 1) for line in
                    (root / "cleanup_status.txt").read_text().splitlines())
    required = ("run_exit", "stop_exit", "stop_verify_exit", "restore_exit",
                "sha_exit", "sha_compare_exit", "script_sha_exit",
                "script_compare_exit", "pre_admission_exit")
    if not set(required).issubset(statuses) or any(statuses[key] != "0" for key in required):
        raise ValueError(f"cleanup did not succeed: {statuses}")
    optional = set(statuses) - set(required)
    if optional not in (set(), {"admission_exit", "final_exit"}) or any(
            statuses[key] != "0" for key in optional):
        raise ValueError(f"unexpected final cleanup status: {statuses}")
    if (root / "source_before.sha256").read_bytes() != (root / "source_after.sha256").read_bytes():
        raise ValueError("borrowed source SHA did not return to original")
    if (root / "scripts_before.sha256").read_bytes() != (root / "scripts_after.sha256").read_bytes():
        raise ValueError("ledger/controller/validator script SHA changed during run")
    client = json.loads((root / "client_admission.json").read_text())
    server = json.loads((root / "server_admission.json").read_text())
    warm = json.loads((root / "warmup_phase_barrier.json").read_text())
    if (client["status"] != "client_two_phase_admitted" or client["request_count"] != 96
            or server["status"] != "server_two_phase_admitted"
            or warm["status"] != "warmup48_phase_barrier_pass"):
        raise ValueError("client/server/barrier admission missing")
    if (root / "server_post_count.txt").read_text().strip() != "96":
        raise ValueError("server POST count mismatch")
    install = json.loads((root / "install.json").read_text())
    restore = json.loads((root / "restore.json").read_text())
    if (install.get("installed") is not True or restore.get("restored") is not True
            or restore.get("helper_sha_drift") is not False
            or install.get("files") != restore.get("files")):
        raise ValueError("patch install/restore record mismatch")
    files = {
        "client_admission.json", "server_admission.json", "warmup_phase_barrier.json",
        "source_before.sha256", "source_after.sha256", "scripts_before.sha256",
        "scripts_after.sha256", "install.json", "restore.json",
        "server_post_count.txt", "phase_transitions.jsonl",
    }
    result = {
        "status": "instrumented_bound_diagnostic_admitted_not_formal_tps",
        "scope": "Host/client lineage only; device freshness and Product Bound unresolved",
        "run_dir": str(root),
        "input_sha256": {name: sha(root / name) for name in sorted(files)},
    }
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": result["status"]}))


if __name__ == "__main__":
    main()
