#!/usr/bin/env python3
"""Bind the diagnostic to native plans, with fresh UID/time checks before launch."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import uuid


def fresh_lease(pod, reservation, now):
    assert pod["metadata"]["name"] == reservation["name"]
    assert pod["metadata"]["labels"]["owner"] == reservation["owner_label"]
    assert pod["spec"]["nodeName"] == reservation["node_name"] and pod["status"]["phase"] == "Running"
    assert int(pod["spec"]["containers"][0]["resources"]["requests"]["huawei.com/Ascend910"]) == 16
    uid = pod["metadata"]["uid"]
    uuid.UUID(uid)
    starts = [pod["metadata"]["creationTimestamp"]]
    if pod["status"].get("startTime"):
        starts.append(pod["status"]["startTime"])
    # API server and node wall clocks differ here. Use the earlier reported
    # timestamp against the Mac clock so a later API creation cannot extend it.
    start = min(dt.datetime.fromisoformat(t.replace("Z", "+00:00")) for t in starts)
    deadline = int(pod["spec"]["activeDeadlineSeconds"])
    assert 0 < deadline <= 21600, "holder must have a finite deadline within the finite research window"
    expiry = start + dt.timedelta(seconds=deadline)
    lifetime = min(14400, int((expiry - now).total_seconds()) - 360)
    assert lifetime >= 900, "insufficient current holder time; renew the owned holder then regenerate"
    return uid, lifetime, expiry.isoformat()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-manifest", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lease32", type=Path)
    p.add_argument("--lease33", type=Path)
    a = p.parse_args()
    assert re.fullmatch(r"[a-z][a-z0-9-]{3,35}", a.run_id)
    assert bool(a.lease32) == bool(a.lease33) and not a.output.exists()
    root = Path(__file__).resolve().parent
    repo = root.parents[2]
    packet = json.loads((repo / "glm5-3-vllm/diagnostics/p-first-state-v1/review-plan.json").read_text())
    build = json.loads(a.build_manifest.read_text())
    overlay = a.build_manifest.with_name("model_runner_v1.py")
    assert hashlib.sha256(overlay.read_bytes()).hexdigest() == build["overlay_sha256"]
    assert build["base_sha256"] == "568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c"
    remote_overlay = ("/data/wio/glm5-3-vllm-execution/20261017-official-pd/execution-patches/"
                      "state-tail-probe-v1/" + build["overlay_sha256"][:12] + "/model_runner_v1.py")
    now = dt.datetime.now(dt.timezone.utc)
    plans = {}
    for role, node, lease in (("P", 32, a.lease32), ("D", 33, a.lease33)):
        base = (repo / packet["base_configs"][role]["path"]).read_bytes()
        assert hashlib.sha256(base).hexdigest() == packet["base_configs"][role]["sha256"], "native launch reference changed"
        plan = json.loads(base)
        assert not plan.get("runtime_patches")
        reservation = plan["reservation"]
        reservation.update(name=f"glm53-state-tail-{node}-{a.run_id}",
                           owner_label="codex-state-tail-" + a.run_id, expected_uid=None)
        plan.update(created_utc=now.isoformat(), objective="Cold C1 native tail-state diagnostic N256/257 only",
                    notes=["No publication/query adoption, full-prefix proof or performance test.",
                           "Existing sustained execution authorization applies. Stop this request batch on first failure, then analyze and continue research.",
                           "P/D each TP16; readiness watchdog15min, bounded model watchdog<=4h, holder<=6h with owned renewal permitted."],
                    schema="glm53-state-tail-launch-template/v1", requires_lifetime_refresh=True, state_tail_probe=True)
        plan.pop("lifetime_refreshed_client_utc", None)
        plan["remote"]["run_dir"] = f"/data/wio/glm5-3-vllm-execution/20261018-state-tail/{a.run_id}/node{node}"
        plan["container"].update(name=f"glm53-state-tail-{node}-{a.run_id}", lifetime_seconds=5400)
        plan["env"].update(GLM53_STATE_TAIL_PROBE="1", GLM53_STATE_PROBE_OUTPUT="/out/state-tail-probe")
        plan["runtime_patches"] = [{"source": remote_overlay, "target": build["target"],
            "sha256": build["overlay_sha256"], "base_sha256": build["base_sha256"]}]
        argv = plan["serve_argv"]
        index = argv.index("--kv-transfer-config") + 1
        kv = json.loads(argv[index]); kv["engine_id"] = f"glm53-state-tail-{role.lower()}-{a.run_id}"
        argv[index] = json.dumps(kv, separators=(",", ":"))
        if lease:
            pod = json.loads(lease.read_text())
            uid, lifetime, expiry = fresh_lease(pod, reservation, now)
            reservation["expected_uid"] = uid
            plan["container"]["lifetime_seconds"] = lifetime
            plan.update(schema="glm53-capacity-launch/v1", requires_lifetime_refresh=False,
                        holder_expiry_utc=expiry, lifetime_refreshed_client_utc=now.isoformat())
        plans[role] = plan
    a.output.mkdir(parents=True)
    for role, plan in plans.items():
        (a.output / (role + ".json")).write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"plans": str(a.output), "launch_schema_bound_to_fresh_leases": bool(a.lease32),
                     "remote_overlay_to_stage_after_window_authorization": remote_overlay,
                     "device_action_performed": False}))


if __name__ == "__main__":
    main()
