"""Read-only reconciliation of a running controller; never restart/replay."""
import hashlib
import json
from pathlib import Path
import sys

J = Path(__file__).parent
R = J.parents[1]
G = R.parents[4]
sys.path.insert(0, str(G / "runtime"))
from phase_runner import atomic_json, same_process, utc
job = json.loads(Path(sys.argv[1]).read_text())
state = json.loads((R / "state.json").read_text())
alive = same_process(state["owner"])
if state["status"] == "running":
    assert alive
snapshot = dict(at=utc(), state=state, controller_alive=alive,
                service_operations=0, generation_requests=0,
                artifacts=[dict(name=p.name, bytes=p.stat().st_size) for p in R.iterdir() if p.is_file()])
atomic_json(J / "snapshot.json", snapshot)
p = J / "snapshot.json"
ev = dict(id="snapshot", path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
          bytes=p.stat().st_size, locator="controller identity/state and artifact inventory; diagnostic not adjudicated")
result = dict(schema_version=1, job_id=job["job_id"], status="completed",
              summary="Read-only controller reconciliation; no resource action or replay.",
              execution=dict(inner_exit_code=0, acceptance="passed", processes=[dict(host="166", role="unique-controller",
                    pid=state["owner"]["pid"], readiness="ready", evidence_ids=["snapshot"])] if alive else []),
              findings=[dict(kind="fact", text="Controller state: " + state["status"] + "; identity alive=" + str(alive),
                             scope=dict(run="GLM-RUN-0242", read_only=True, performance_verdict=False), evidence_ids=["snapshot"])],
              evidence=[ev], unknowns=["Ready/commit causality, full E2E Gain and KEEP remain unadjudicated"],
              decision_request=None, next_check_at=None)
atomic_json(Path(job["result"]["path"]), result)
