#!/usr/bin/env python3
"""Source-interface, real CPU view and diagnostic-rejection checks; no fake torch."""
import ast
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    root = Path(__file__).resolve().parent
    probe, comparer = load("tail_probe", root / "probe.py"), load("tail_compare", root / "compare.py")
    repo = root.parents[2]
    packet = json.loads((repo / "glm5-3-vllm/diagnostics/p-first-state-v1/review-plan.json").read_text())
    sources = {r["path"]: Path(r["local_path"]) for r in packet["source_identity"]}
    checks = {}
    client = load("tail_client", root / "run_requests.py")
    for role, ref in packet["base_configs"].items():
        argv = json.loads((repo / ref["path"]).read_text())["serve_argv"]
        assert argv[argv.index("--served-model-name") + 1] == client.MODEL
    assert [c["prompt_sha256"] for c in client.CASES] == [c["prompt_sha256"] for c in packet["cases"]]
    checks["client_model_and_prompt_identity_match_native_plans"] = True
    for source in root.glob("*.py"):
        compile(source.read_text(), str(source), "exec")
    # Check exact signatures/field names against the now image-matched source.
    def definition(path, cls, name):
        tree = ast.parse(sources[path].read_text())
        owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
        return next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)
    bucket = definition("vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/pull_worker.py",
                        "MooncakePullRecvingThread", "_execute_tp_transfer_bucket")
    assert [a.arg for a in bucket.args.args] == ["self", "remote_pp_rank", "remote_pcp_rank", "remote_tp_rank",
        "remote_pcp_size", "remote_tp_size", "remote_dcp_size", "remote_metadata", "transfer_entries_by_spec"]
    draft = definition("vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py",
                       "AscendSpecDecodeBaseProposer", "_run_merged_draft")
    assert {"num_tokens", "batch_size", "multi_steps_attn_metadata"}.issubset({a.arg for a in draft.args.args})
    checks["native_source_interfaces"] = True
    # Aggregator construction is outside the 18 image-matched files. Keep
    # these three pinned official sources explicitly reference-only; runtime
    # acceptance still requires all 16 actual P rank records.
    refs = json.loads((repo / "glm5-3-vllm/evidence/20261018-p-first-state-audit/state-tail-probe-v1/aggregator-source-references.json").read_text())
    ref_text = {}
    for ref in refs:
        data = Path(ref["local_path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == ref["sha256"]
        assert ref["grade"] == "fixed_commit_reference_only" and ref["image_sha_match"] is False
        ref_text[ref["path"]] = data.decode()
    core = ast.parse(ref_text["vllm/v1/engine/core.py"])
    guard = next(n for n in ast.walk(core) if isinstance(n, ast.If) and
                 ast.unparse(n.test) == "self.scheduler.connector is not None" and
                 "init_kv_output_aggregator" in ast.unparse(n))
    assert len(guard.body) == 1 and isinstance(guard.body[0], ast.Expr)
    assert ast.unparse(guard.body[0].value) == "self.model_executor.init_kv_output_aggregator(self.scheduler.connector)"
    utils = ast.parse(ref_text["vllm/distributed/kv_transfer/kv_connector/utils.py"])
    owner = next(n for n in utils.body if isinstance(n, ast.ClassDef) and n.name == "KVOutputAggregator")
    creator = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "from_connector")
    assert len(creator.body) == 1 and ast.unparse(creator.body[0]) == "return cls(connector.get_finished_count() or world_size)"
    checks["fixed_reference_aggregator_includes_producer_not_device_proof"] = True
    tree = ast.parse((root / "probe.py").read_text())
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and
                   n.func.attr in ("synchronize", "barrier", "wait_event", "wait_stream", "clone") for n in ast.walk(tree))
    checks["no_new_sync_or_clone_call"] = True
    with tempfile.TemporaryDirectory(prefix="glm53-overlay-build-") as tmp:
        output = Path(tmp) / "build"
        subprocess.check_output([sys.executable, str(root / "build.py"), "--base",
            str(sources["vllm-ascend/vllm_ascend/worker/model_runner_v1.py"]), "--output", str(output)])
        built = ast.parse((output / "model_runner_v1.py").read_text())
        native = ast.parse(sources["vllm-ascend/vllm_ascend/worker/model_runner_v1.py"].read_text())
        assert all(ast.dump(a) == ast.dump(b) for a, b in zip(native.body, built.body))
        gate = ast.Module(body=[built.body[-1]], type_ignores=[])
        old_gate = os.environ.pop("GLM53_STATE_TAIL_PROBE", None)
        namespace = {"__file__": str(output / "model_runner_v1.py")}
        try:
            exec(compile(gate, "<default-off-gate>", "exec"), namespace)
        finally:
            if old_gate is not None:
                os.environ["GLM53_STATE_TAIL_PROBE"] = old_gate
        assert "_glm53_probe" not in namespace
        checks["native_definitions_unchanged_and_default_gate_off"] = True
        output_http = Path(tmp) / "must-not-exist"
        attempted = subprocess.run([sys.executable, str(root / "run_requests.py"),
            "--p-url", "http://172.16.12.32:29481", "--d-url", "http://172.16.12.33:29482",
            "--output", str(output_http), "--p-probe-dir", "/data/wio/glm5-3-vllm-execution/20261018-state-tail/fixture/node32/state-tail-probe"], capture_output=True)
        assert attempted.returncode != 0 and not output_http.exists()
        checks["functional_client_default_gate_refuses_before_IO"] = True

    prepare = load("tail_prepare", root / "prepare_launch.py")
    now = dt.datetime(2026, 10, 18, tzinfo=dt.timezone.utc)
    reservation = {"name": "fixture-fresh", "owner_label": "fixture-owner", "node_name": "node32"}
    pod = {"metadata": {"name": "fixture-fresh", "uid": "11111111-2222-4333-8444-555555555555",
                       "labels": {"owner": "fixture-owner"}, "creationTimestamp": "2026-10-18T00:00:00Z"},
           "spec": {"nodeName": "node32", "activeDeadlineSeconds": 1500,
                    "containers": [{"resources": {"requests": {"huawei.com/Ascend910": 16}}}]},
           "status": {"phase": "Running"}}
    assert prepare.fresh_lease(pod, reservation, now)[1] == 1140
    for label, mutate in [("stale_holder_name", lambda p: p["metadata"].update(name="old-holder")),
                          ("wrong_holder_node", lambda p: p["spec"].update(nodeName="node33")),
                          ("expired_holder", lambda p: p["spec"].update(activeDeadlineSeconds=500))]:
        bad = copy.deepcopy(pod); mutate(bad)
        try:
            prepare.fresh_lease(bad, reservation, now)
        except AssertionError:
            checks[label + "_rejected"] = True
        else:
            raise AssertionError(label)
    start_tree = ast.parse((repo / "glm5-3-vllm/tools/start_capacity.py").read_text())
    uid_guard = next(n for n in ast.walk(start_tree) if isinstance(n, ast.If)
                     and ast.unparse(n.test) == "'expected_uid' in res")
    compiled_guard = compile(ast.Module(body=[uid_guard], type_ignores=[]), "<native-launcher-uid-guard>", "exec")
    guard_context = {"res": {"expected_uid": "fresh"}, "datetime": dt,
        "plan": {"container": {"lifetime_seconds": 900}, "state_tail_probe": True},
        "pod": {"metadata": {"uid": "fresh", "creationTimestamp": dt.datetime.now(dt.timezone.utc).isoformat()},
                "spec": {"activeDeadlineSeconds": 1500}}}
    exec(compiled_guard, guard_context)
    try:
        exec(compiled_guard, {"res": {"expected_uid": "fresh"}, "pod": {"metadata": {"uid": "stale"}}})
    except AssertionError:
        checks["actual_launcher_UID_guard_rejects_replacement"] = True
    else:
        raise AssertionError("UID guard")
    stale = copy.deepcopy(guard_context["pod"])
    stale["metadata"]["creationTimestamp"] = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=10)).isoformat()
    try:
        exec(compiled_guard, {**guard_context, "pod": stale})
    except AssertionError:
        checks["actual_launcher_rejects_stale_lifetime"] = True
    else:
        raise AssertionError("lifetime guard")

    # Real ndarray storage/strides mirror the native prescribed 4D views.
    # Widths here are schema fixtures, NOT observed runtime cache dimensions.
    main_spec = {"type": "AscendMLAAttentionSpec", "block_size": 128, "heads": 1,
                 "head_size": 656, "scale_dim": 0, "packed_c8": True}
    page_stride = 128 * 656 + 64
    backing = np.zeros(5 * page_stride, dtype=np.uint8)
    packed = np.ndarray((5, 128, 1, 656), np.int8, backing,
                        strides=(page_stride, 656, 656, 1))
    for pos in (254, 255, 256):
        block, offset = {254: (1, 126), 255: (1, 127), 256: (4, 0)}[pos]
        assert probe.row_location(packed.shape, packed.strides, main_spec, block, offset, 0) == (block, offset)
        packed[block, offset] = (pos % 127)
        assert len(packed[block, offset].tobytes()) == 656
    checks["packed_page_strided_view_256_257_boundary"] = True
    k_bytes, scale_bytes = 5 * 128 * 128, 5 * 128 * 2
    storage = np.zeros(k_bytes + scale_bytes, dtype=np.uint8)
    key = np.ndarray((5, 128, 1, 128), np.int8, storage)
    scale = np.ndarray((5, 128, 1, 1), np.float16, storage, offset=k_bytes)
    indexer = {"type": "AscendSFAIndexerCacheSpec", "block_size": 128, "heads": 1,
               "head_size": 128, "scale_dim": 1, "replication": 1}
    for c, value in enumerate((key, scale)):
        stride = tuple(s // value.itemsize for s in value.strides)
        assert probe.row_location(value.shape, stride, indexer, 4, 0, c) == (4, 0)
    checks["split_indexer_K_scale_shared_storage_views"] = True
    for label, args in [
        ("transposed_shape", ((5, 1, 128, 656), packed.strides, main_spec, 1, 0, 0)),
        ("wrong_stride", (packed.shape, (page_stride, 657, 656, 1), main_spec, 1, 0, 0)),
        ("missing_block", (packed.shape, packed.strides, main_spec, 5, 0, 0)),
        ("wrong_scale_width", (key.shape, key.strides, indexer, 1, 0, 1)),
    ]:
        try:
            probe.row_location(*args)
        except ValueError:
            checks[label + "_rejected"] = True
        else:
            raise AssertionError(label)

    # Pairing fixtures exercise rejection only; never stand in for live traces.
    p, d, requests = [], [], []
    layers = [{"name": "target", "views": [{}]}, {"name": "mtp", "views": [{}]},
              {"name": "indexer", "views": [{}, {}]}]
    p.extend([{"kind": "runner_identity", "tp_rank": 0, "runnable_hook_bound": True},
              {"kind": "cache_registry", "tp_rank": 0, "layers": layers}])
    d.append({"kind": "cache_registry", "tp_rank": 0, "layers": layers})
    for n in (256, 257):
        remote = "fixture-P-" + str(n)
        requests.append({"input_tokens": n, "prompt_sha256": "fixture", "functional_pass": True, "P_response": {
            "kv_transfer_params": {"remote_request_id": remote}}})
        rows = []
        for layer in layers:
            for component in range(len(layer["views"])):
                for pos in (n - 2, n - 1):
                    rows.append({"layer": layer["name"], "family": "MTP" if layer["name"] == "mtp" else "target",
                        "component": component, "position": pos, "block": 4, "offset": pos % 128,
                        "writer_slot": 4 * 128 + pos % 128, "writer_metadata_observed": True,
                        "dtype": "torch.int8", "row_shape": [1, 4], "bytes": 4,
                        "sha256": probe.row_digest("torch.int8", (1, 4), b"\0" * 4)})
        p.append({"kind": "P_tail", "tp_rank": 0, "req_id": remote, "n": n, "prompt_sha256": "fixture",
            "status": "observed", "boundary": "native_get_output_returned", "mtp_calls": 1, "scheduled_K": 1,
            "runnable_hook_bound": True,
            "event_ready_after_native_wait": True, "mtp_stream": 7, "constructor_stream": 7,
            "mtp_layer_names": ["mtp"], "rows": rows, "refused_layers": []})
        drows = [{**r, "remote_tp": 0, "remote_block": 4, "block": 2} for r in rows]
        d.append({"kind": "D_tail", "tp_rank": 0, "remote_req_id": remote, "n": n, "status": "observed",
            "boundary": "native_handle_requests_returned_before_finished_enqueue", "rows": drows,
            "refused_layers": []})
    good = comparer.compare(p, d, requests, ranks=(0,))
    assert good["verdict"] == "sampled_rows_match" and good["full_prefix_proved"] is False
    checks["finite_pair_fixture_never_proves_prefix"] = True
    assert comparer.compare([], d, requests)["verdict"] == "incomplete"
    checks["empty_records_rejected"] = True
    for label, mutate in [
        ("wrong_source_rank", lambda a, b: b[1]["rows"][0].update(remote_tp=1)),
        ("wrong_source_block", lambda a, b: b[1]["rows"][0].update(remote_block=3)),
        ("byte_mismatch", lambda a, b: b[1]["rows"][0].update(sha256="wrong")),
        ("missing_component", lambda a, b: b[1]["rows"].pop()),
        ("late_D_boundary", lambda a, b: b[1].update(boundary="after_D_query")),
        ("no_writer_provenance_even_equal_zero_bytes", lambda a, b: a[2]["rows"][0].update(writer_metadata_observed=False)),
        ("wrong_writer_slot", lambda a, b: a[2]["rows"][0].update(writer_slot=0)),
        ("no_P_mtp_call", lambda a, b: a[2].update(mtp_calls=0)),
        ("missing_native_completion", lambda a, b: a[2].update(event_ready_after_native_wait=False)),
    ]:
        cp, cd = copy.deepcopy(p), copy.deepcopy(d)
        mutate(cp, cd)
        assert comparer.compare(cp, cd, requests, ranks=(0,))["verdict"] != "sampled_rows_match", label
        checks[label + "_rejected"] = True
    assert all(checks.values())
    print(json.dumps({"checks": checks, "device_run": False, "full_native_import": False,
        "scope": "Matched source interface, actual NumPy views/strides and synthetic pairing records only; no torch, model or device replay",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
