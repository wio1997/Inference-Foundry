#!/usr/bin/env python3
"""Run every R247 CPU suite and record exit codes plus source SHA."""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys

R = pathlib.Path(__file__).resolve().parent.parent
SUITES = ["check_static_cpu.py", "check_source_delta_cpu.py",
          "check_launcher_path_cpu.py", "check_r246_gate_cpu.py",
          "check_profile_collection_cpu.py", "check_controller_status_cpu.py",
          "check_launch_closure_cpu.py"]


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main():
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    results = {}
    for name in SUITES:
        path = R / "checks" / name
        q = subprocess.run([sys.executable, "-B", str(path)], capture_output=True,
                           text=True, timeout=300)
        results[name] = {"rc": q.returncode, "stdout_tail": q.stdout[-6000:],
                         "stderr_tail": q.stderr[-6000:], "sha256": sha(path)}
        print(name, q.returncode)
        if q.returncode != 0:
            print(q.stdout[-2000:])
            print(q.stderr[-2000:])
    out = {"project_id": "glm53-flash", "unit": "r247",
           "kind": "CPU_TEST_EVIDENCE_NO_MODEL_CALL",
           "passed": all(row["rc"] == 0 for row in results.values()),
           "suites": results,
           "no_model_HTTP_NPU_calls": True}
    target = R / "results/cpu-tests.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2) + "\n")
    print("passed", out["passed"])
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
