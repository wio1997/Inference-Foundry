#!/usr/bin/env python3
"""Build hash-pinned Run401 manifest only after clean A0/B/A1 gates."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "evidence/20260927_loop078_bound"
RUNS = {"A0": 413, "B": 410, "A1": 417}


def ref(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    timing_path = BASE / "run401/timing_preflight.json"
    timing = json.loads(timing_path.read_text())
    assert timing["passed"] and timing["devices"] == list(range(8)) and timing["trials"] == 32
    controls = {}
    evidence = []
    for arm, run in RUNS.items():
        d = BASE / f"run{run}"
        gate_path = d / "gate.json"
        gate = json.loads(gate_path.read_text())
        assert gate["arm"] == arm and gate["http_posts"] == 60 and gate["source_restored"]
        assert (d/"source_before.sha256").read_bytes() == (d/"source_after.sha256").read_bytes()
        controls[arm] = gate
        for name in ("gate.json", "source_before.sha256", "source_after.sha256", "server_post_count.txt"):
            evidence.append(ref(d / name))
        if arm == "B":
            assert len(list((d/"capture").glob("rank*_cohort*.json"))) == 40
            for name in ("install.json", "restore.json", "cleanup_status.txt"):
                evidence.append(ref(d / name))
    manifest = {"timing_report": ref(timing_path), "evidence_files": evidence,
        "controls": controls, "intervening_container_restart": True,
        "interpretation": "A0/A1 do not retain per-cycle acceptance; original-schedule extrapolation gate must fail, while local B event observations can be evaluated."}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "manifest_pinned", "evidence_files": len(evidence)}))


if __name__ == "__main__":
    main()
