"""Dispatch exactly one immutable stage to the existing resource controller."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

J = Path(__file__).parent
job = json.loads(Path(sys.argv[1]).read_text())
R = J.parents[1]
G = R.parents[4]
sys.path.insert(0, str(G / "runtime"))
from phase_runner import atomic_json, same_process

assert not (R / "state.json").exists(), "never replay a Run"
for item in job["inputs"]:
    if item.get("sha256"):
        assert hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"]
p = subprocess.run(["/usr/bin/python3", str(G / "runtime/controller.py"), "start",
                    str(R / "controller_spec.json")], capture_output=True, timeout=30)
(J / "launch.stdout").write_bytes(p.stdout)
(J / "launch.stderr").write_bytes(p.stderr)
state = json.loads((R / "state.json").read_text()) if (R / "state.json").exists() else {}
atomic_json(J / "dispatch_state.json", state)
ok = p.returncode == 0 and state.get("status") == "running" and same_process(state["owner"])
code = 0 if ok else 1
evidence = []
for name in ("launch.stdout", "launch.stderr", "dispatch_state.json"):
    path = J / name
    evidence.append(dict(id=name, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         bytes=path.stat().st_size, locator="entire dispatch output; not diagnostic completion"))
result = dict(schema_version=1, job_id=job["job_id"], status="running" if ok else "failed",
              summary="Existing unique controller dispatched one H1 observational diagnostic; results pending.",
              execution=dict(inner_exit_code=code, acceptance="passed" if ok else "failed",
                             controller_start_exit_code=p.returncode,
                             processes=[dict(host="166", role="unique-controller", pid=state["owner"]["pid"],
                                             readiness="ready", evidence_ids=["dispatch_state.json"])] if ok else []),
              findings=[dict(kind="fact", text="Controller running, exact Run spec pinned; no completion claim.",
                             scope="dispatch only", evidence_ids=["dispatch_state.json"])] if ok else [],
              evidence=evidence, unknowns=["Diagnostic correctness, ready/commit causality and restoration pending; E2E Gain unknown"],
              decision_request=None, next_check_at=None)
atomic_json(Path(job["result"]["path"]), result)
sys.exit(code)
