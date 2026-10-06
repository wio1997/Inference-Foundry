"""Reduce one finished trace window; no controller or service mutations."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

J = Path(__file__).parent
R = J.parents[1]
G = R.parents[4]
S = G / "records/points/GLM-OPT-0002/research/engine_commit_20261006"
sys.path.insert(0, str(G / "runtime"))
from phase_runner import atomic_json, same_process
job = json.loads(Path(sys.argv[1]).read_text())
root = json.loads((R / "trace_root.json").read_text())
assert not same_process(dict(pid=root["pid"], **root["identity"])), "only reduce a stopped trace epoch"
assert (R / "client_summary.json").exists() or (R / "diagnostic_error.json").exists()
for source in job["inputs"]:
    if source.get("sha256"):
        assert hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() == source["sha256"]
a = time.monotonic_ns()
wall = time.time_ns()
b = time.monotonic_ns()
atomic_json(J / "clock_calibration.json", dict(monotonic_before_ns=a, realtime_ns=wall, monotonic_after_ns=b,
                                              scope="post trace, no claim of NTP stability or profiler-clock equivalence"))
p = subprocess.run(["/usr/bin/python3", str(S / "reduce_commit_trace.py"), str(R), "--output", str(R / "reduction")],
                   capture_output=True, timeout=120)
(J / "reduce.stdout").write_bytes(p.stdout)
(J / "reduce.stderr").write_bytes(p.stderr)
ev = []
for name, path in [("stdout", J / "reduce.stdout"), ("stderr", J / "reduce.stderr"),
                   ("clock", J / "clock_calibration.json"), ("summary", R / "reduction/summary.json"),
                   ("decisive", R / "reduction/decisive.json")]:
    if path.exists():
        ev.append(dict(id=name, path=str(path), bytes=path.stat().st_size,
                       sha256=hashlib.sha256(path.read_bytes()).hexdigest(), locator="mechanical reduction; Sol must inspect referenced raw"))
ok = p.returncode == 0
summary = json.loads((R / "reduction/summary.json").read_text()) if ok else {}
atomic_json(Path(job["result"]["path"]), dict(schema_version=1, job_id=job["job_id"],
    status="completed" if ok else "failed", summary="Mechanical trace correlation only; no performance verdict.",
    execution=dict(inner_exit_code=p.returncode, acceptance="passed" if ok else "failed", processes=[]),
    findings=[dict(kind="fact", text=str(summary.get("proven_ready_before_extra_schedule_cases")) + " proven early-publication cases / " + str(summary.get("enqueue_before_older_commit_cases")) + " older-commit cases",
                   scope=dict(run="GLM-RUN-0242", device_idle_proven=False, performance_verdict=False), evidence_ids=["summary", "decisive"])] if ok else [],
    evidence=ev, unknowns=["Legal commit/output causality, device-clock alignment and true complete E2E Gain require Sol review"],
    decision_request=None, next_check_at=None))
sys.exit(p.returncode)
