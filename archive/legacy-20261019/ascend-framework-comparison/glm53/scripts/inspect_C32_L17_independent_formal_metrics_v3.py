"""DRAFT_PENDING_PRIMARY_REVIEW: read the completed L17 CPU43 formal archive (v2).

This is not a model run.  It re-streams the original formal D metric archive
from the completed CPU43 controller run, independently re-derives all native
bodies, DP gauges and per-rank Graph counters, and pins those values to the new
L17 client audit.  One SSH, no model/SDK/profile action.
"""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import glm53_C32_L17_pipeline_contract_v4 as contract

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = contract.INDEPENDENT_METRICS_RUN
FORMAL = contract.FORMAL_RUN
CLIENT_AUDIT = contract.FORMAL_CLIENT_AUDIT
OLD_CONTROL_SOCKET = "/Users/wio/work/sglang-glm53-pd/ssh-control"
FRESH_CONTROL_SOCKET = "/Users/wio/work/sglang-glm53-pd/ssh-control-l17-recovery-20261019"


def load(name, relative, sha):
    path = ROOT / relative
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plan(old=None):
    if old is None:
        old = load("l17_capture_base", contract.OLD_CAPTURE_BASE, contract.OLD_CAPTURE_BASE_SHA)
    mixed = load("l17_capture_core", "scripts/glm53_C32_L17_native_capture_v5.py", "68b88d8a1d13ba9cff10037b858f61d7d5290d5091b026b11a7b879cc1439424")
    client = json.loads((ROOT / CLIENT_AUDIT).read_text())
    assert client["client_audit_pass"] is True
    code = old.render(old.CPU_HOST, {
        "ARCHIVE": "/data/wio/inference-foundry/glm53/runs/" + RUN_ID,
        "FORMAL_RID": FORMAL,
        "CONCURRENCY": 32,
    })
    code += "\nassert out['formal_report_identity']['sha256']==" + repr(client["raw_formal_report_sha256"]) + "\n"
    code += "assert out['independent_formal_metric_archive_audit']['archive_sha256']==" + repr(client["D_metric_archive_sha256"]) + "\n"
    ast.parse(code)
    return old, mixed, code


def main():
    assert Path(FRESH_CONTROL_SOCKET).is_socket(), "fresh SSH control socket missing: " + FRESH_CONTROL_SOCKET
    old = load("l17_capture_base", contract.OLD_CAPTURE_BASE, contract.OLD_CAPTURE_BASE_SHA)
    assert len(old.base.SSH) > 2 and old.base.SSH[2] == OLD_CONTROL_SOCKET, "unexpected inventory SSH control socket"
    old.base.SSH = list(old.base.SSH)
    old.base.SSH[2] = FRESH_CONTROL_SOCKET
    rpc = json.loads((ROOT / "runs" / FORMAL / "controller-rpc-state.json").read_text())
    assert rpc["done_received"] and rpc["connection_rc"] is not None
    old, mixed, code = plan(old)
    review = json.loads((ROOT / contract.INDEPENDENT_METRICS_REVIEW).read_text())
    assert review["host_code_sha256"] == hashlib.sha256(code.encode()).hexdigest()
    run = ROOT / "runs" / RUN_ID
    run.mkdir(parents=True, exist_ok=False)
    (run / "execution-once.lock").write_text(str(os.getpid()))
    receipt = {
        "project_id": "glm53",
        "run_id": RUN_ID,
        "formal_run_id": FORMAL,
        "model_requests": 0,
        "SDK_init_calls": 0,
        "max_SSH": 1,
        "only_completed_CPU_archive_read": True,
        "profiling": False,
    }
    try:
        wire = mixed._invoke_with_raw_failure(old, 43, code, "/data/wio/inference-foundry/glm53/runs/" + RUN_ID, run)
        (run / "node-43.json").write_text(json.dumps(wire, indent=2) + "\n")
        receipt["node_rc"] = wire["rc"]
        assert wire["rc"] == 0
        data = json.loads(wire["stdout"])
        audit = data["independent_formal_metric_archive_audit"]
        assert audit["all_native_bodies_SHA_and_DP_gauges_recomputed"] is True
        assert not audit["errors"] and audit["sample_pairs"] == audit["complete_pairs"]
        assert data["formal_native_target_graph_counters"]["delta_by_global_EP_rank"]
        assert set(data["formal_native_target_graph_counters"]["delta_by_global_EP_rank"]) == {str(index) for index in range(32)}
        assert all(value > 0 for value in data["formal_native_target_graph_counters"]["delta_by_global_EP_rank"].values())
        receipt["sample_pairs"] = audit["sample_pairs"]
        receipt["native_endpoint_rows"] = audit["native_HTTP_rows"]
        receipt["all32productive_Graph_deltas"] = True
        receipt["pass"] = True
    finally:
        (run / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
