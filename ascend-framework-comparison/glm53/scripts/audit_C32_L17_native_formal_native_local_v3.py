"""DRAFT_PENDING_PRIMARY_REVIEW: independent L17 formal native audit (v2).

Joins the original formal client report to the post-formal native capture, the
fresh pre-formal v2 native baseline capture and the independent CPU43 metric
archive.  It
requires all32 P native cache readers, all32 typed-SDMA D destinations, all32
productive Graph ranks, strict SDK begin==good/bad0/inflight0 plus the 11th
TOTAL column, MTP5, and the source-qualified optional D storage review with raw
nulls preserved.  No model, SDK, SSH or resource call is made.
"""
import argparse
import collections
import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path

import glm53_C32_L17_pipeline_contract_v4 as contract

ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    return json.loads((ROOT / relative).read_text())


def sha(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def unpack(capture, node):
    relative = f"runs/{capture}/node-{node}.json"
    _, _, obj = contract.load_native_capture(relative)
    events = contract.persistent_events(obj)
    return relative, obj, events


def sdk_counts(obj, pids):
    result = {}
    for snapshot in obj["native_SDK_small_file_snapshots"]:
        assert snapshot["rc"] == 0
        if snapshot["container_id"] != obj["container_id"]:
            continue
        for item in snapshot["files"]:
            assert item["complete_line_ending"]
            assert hashlib.sha256(item["text"].encode()).hexdigest() == item["sha256"]
            match = re.search(r"ptracer_(\d+)\.dat$", item["path"])
            if not match or int(match.group(1)) not in pids:
                continue
            pid = int(match.group(1))
            tags = {}
            for line in item["text"].splitlines():
                parts = line.split()
                if len(parts) > 11 and parts[2] in [
                    "TP_MMC_PY_BATCH_GET",
                    "TP_MMC_LOCAL_BATCH_GET",
                    "TP_MMC_LOCAL_BATCH_GET_SIZE",
                    "TP_HYBM_SDMA_BATCH_LD_TO_GD",
                ]:
                    entry = tags.setdefault(parts[2], {
                        "begin": 0, "good": 0, "bad": 0, "on_fly_last": None,
                        "raw_TOTAL_column_sum": Decimal(0),
                    })
                    entry["begin"] += int(parts[3])
                    entry["good"] += int(parts[4])
                    entry["bad"] += int(parts[5])
                    entry["on_fly_last"] = int(parts[6])
                    entry["raw_TOTAL_column_sum"] += Decimal(parts[11])
            for tag, entry in tags.items():
                assert entry["begin"] == entry["good"] and entry["bad"] == 0 and entry["on_fly_last"] == 0
                assert entry["raw_TOTAL_column_sum"] >= 0
            result[pid] = tags
    return result


def audit(run_id, capture_id, before_id):
    assert run_id == contract.FORMAL_RUN
    assert capture_id == contract.FORMAL_CAPTURE
    assert before_id == contract.FORMAL_BASELINE_CAPTURE, "before capture must be the fresh pre-formal v2 baseline"
    report = read(f"runs/{run_id}/formal-result.remote.json")
    concurrency = report["concurrency"]
    assert report["framework"] == "sglang" and concurrency == 32
    settings = read(contract.FORMAL_SETTINGS)
    assert settings["run_id"] == run_id
    cache_review = contract.load_module(
        "l17_cache_annotation", contract.CACHE_ANNOTATION_HELPER, contract.CACHE_ANNOTATION_HELPER_SHA
    )
    reviewed = cache_review.review(report)
    assert reviewed["pass"]
    client_path = contract.FORMAL_CLIENT_AUDIT
    client = read(client_path)
    assert client["client_audit_pass"] and client["raw_formal_report_sha256"] == sha(f"runs/{run_id}/formal-result.remote.json")
    functional = read(contract.SHORT_RESULT)
    eager = functional["D_eager_input_acceptance"]
    assert len(eager["worker_rows"]) == 32
    assert len({(row["node"], row["namespace_pid"]) for row in eager["worker_rows"]}) == 32
    assert all(row["native_no_copy_get"] and row["clone_distinct"] for row in eager["worker_rows"])
    assert all(row["is_draft_worker"] is True and row["forward_mode"] == "DECODE" and row["batch_size"] == 2 for row in eager["worker_rows"])
    assert all(row["out_cache_loc"] for row in eager["worker_rows"])
    sources = {
        client_path: sha(client_path),
        contract.SHORT_RESULT: sha(contract.SHORT_RESULT),
        contract.FORMAL_SETTINGS: sha(contract.FORMAL_SETTINGS),
        contract.CACHE_ANNOTATION_HELPER: contract.CACHE_ANNOTATION_HELPER_SHA,
    }
    null_counts = {"P": 0, "D": 0}
    productive_graph = {}
    installation_only_pids = {}
    objects = {}
    runtime = {}
    writes = []
    native_bytes = 0
    read_workers = []
    errors = {}
    sessions = {}
    mtp = []
    tiers = {"P": collections.Counter(), "D": collections.Counter()}
    storage_backends = {"P": collections.Counter(), "D": collections.Counter()}
    for row in report["formal"]["rows"]:
        result = row["result"]
        gateway = result["terminal"]["gateway"]
        extension = result["native"]["extra_details_data"]["glm53_native_sglext"]
        spec = extension["spec_tokens_details"]
        verify = spec["spec_verify_ct"]
        proposed = spec["spec_num_proposed_drafts"]
        correct = spec["spec_num_correct_drafts"]
        histogram = spec["spec_correct_drafts_histogram"]
        assert verify > 0 and proposed == 5 * verify
        assert 0 <= correct <= proposed and len(histogram) == 6
        assert all(type(value) is int and value >= 0 for value in histogram)
        assert sum(histogram) == verify
        assert sum(index * value for index, value in enumerate(histogram)) == correct
        mtp.append({"verify": verify, "proposed": proposed, "correct": correct})
        for role, selected in [("P", gateway["prefill_sglext"]), ("D", extension)]:
            details = selected["cached_tokens_details"]
            numeric = {key: details[key] for key in ["device", "host", "storage"] if key in details}
            annotations = {key: value for key, value in details.items() if key not in numeric}
            tiers[role].update(numeric)
            null_counts[role] += int("storage" not in details)
            if "storage_backend" in annotations:
                storage_backends[role][str(annotations["storage_backend"])] += 1
    for node in contract.ALL_NODES:
        path, obj, events = unpack(capture_id, node)
        before_path, previous, old = unpack(before_id, node)
        sources[path] = sha(path)
        sources[before_path] = sha(before_path)
        objects[node] = obj
        runtime[str(node)] = obj["container_id"]
        expected = settings["runtime_containers"][str(node)]["container_id"]
        assert obj["container_id"] == expected
        assert obj["provider_phase"] == "READY" and obj["/health"]["status"] == 200
        assert {row["namespace_pid"] for row in obj["owned_npu_process_rows"]} == (
            {obj["supervisor"]["provider_pid"], obj["supervisor"]["model_pid"]}
            | {row["pid"] for row in events if row["event"] == "initialize_complete"}
        )
        assert sorted(obj["profiler_artifacts"], key=lambda item: item["path"]) == sorted(previous["profiler_artifacts"], key=lambda item: item["path"])
        assert obj["actual_model_identity"]["namespace_pid"] == obj["supervisor"]["model_pid"]
        assert obj["supervisor"]["phase"] == "MODEL_RUNNING" and "model_exit_code" not in obj["supervisor"]
        for body in [obj["/metrics"], previous["/metrics"]]:
            assert hashlib.sha256(body["body"].encode()).hexdigest() == body["sha256"]
        gauges = [
            float(line.rsplit(" ", 1)[1]) for line in obj["/metrics"]["body"].splitlines()
            if line.startswith(("sglang:num_running_reqs{", "sglang:num_queue_reqs{"))
        ]
        assert gauges and all(value == 0 for value in gauges)
        assert all(log["full_archive_complete"] for log in obj["logs"].values())
        assert all(log["native_sdma"]["copy_calls"] == 0 or log["native_sdma"]["copy_status_values"] == [0] for log in obj["logs"].values())
        init = {row["pid"]: row for row in events if row["event"] == "initialize_complete"}
        assert len(init) == 16
        assert all(row["rc"] == 0 and row["protocol"] == "SDMA" and row["protocol_value"] == 1 for row in init.values())
        sessions.update({row["session"]: (node, row["device_id"]) for row in init.values()})
        marker_lines = [line for log in obj["logs"].values() for line in log["metadata_marker_lines"]]
        assert not any(
            "GLM53_DEEPEP_METADATA_BATCH_FIRST_USE=" in line
            or "GLM53_C32_PACKED_DISPATCH_FIRST_USE=" in line
            or "GLM53_C32_ALIGNMENT_FATAL" in line
            for line in marker_lines
        )
        aligned = contract.alignment_events(obj)
        if node in contract.D_NODES:
            all_alignment_pids = {row["pid"] for row in aligned}
            assert set(init) <= all_alignment_pids
            extras = all_alignment_pids - set(init)
            assert all(
                row["event"] in ["ops_compat_installed", "target_graph_audit_installed", "draft_compat_installed"]
                for row in aligned if row["pid"] in extras
            )
            installation_only_pids[str(node)] = {
                str(pid): dict(collections.Counter(row["event"] for row in aligned if row["pid"] == pid))
                for pid in sorted(extras)
            }
            older_aligned = contract.alignment_events(previous)
            for pid in init:
                rows = [row for row in aligned if row["pid"] == pid]
                assert all(row["revision"] == "glm53-C32-target-graph-fullmesh-v2" for row in rows)
                for method in ["_npu_low_latency_dispatch", "_npu_low_latency_combine"]:
                    first = [row for row in rows if row["event"] == "ops_first_return" and row["method"] == method]
                    assert len(first) == 1 and first[0]["group_size"] == 32 and first[0]["comm_alg"] == "fullmesh_v1"
                    if method.endswith("dispatch"):
                        assert first[0]["quant_mode"] == 2 and first[0]["inputs"]["topk_weights"] is None
                        assert first[0]["outputs"][0]["dtype"] == "torch.int8"
                    else:
                        assert first[0]["comm_quant_mode"] == 0
                        assert first[0]["inputs"]["topk_weights"]["shape"] == [32, 8]
                        assert first[0]["outputs"]["dtype"] == "torch.bfloat16"
                assert any(row["event"] == "draft_graph_disabled" for row in rows)
                assert any(
                    row["event"] == "target_graph_decision" and row["eligible"] is True
                    and row["mode"] == "TARGET_VERIFY" and row["raw_bs"] > 0
                    for row in rows
                )
                assert not any(row["event"] == "target_graph_replay_return" for row in rows)
                older_rows = [row for row in older_aligned if row["pid"] == pid]
                assert rows[:len(older_rows)] == older_rows, "Original per-PID append-only audit prefix must remain intact"
                new_rows = rows[len(older_rows):]
                decisions = [row for row in new_rows if row["event"] == "target_graph_decision"]
                productive = [
                    row for row in decisions
                    if row["eligible"] is True and row["mode"] == "TARGET_VERIFY" and row["raw_bs"] > 0 and not row["is_draft"]
                ]
                assert productive, "Fresh productive target Graph eligibility required on every actual SDK worker"
                rank = next(row["rank"] for row in rows if row["event"] == "ops_first_return")
                assert str(rank) not in productive_graph
                productive_graph[str(rank)] = {
                    "pid": pid,
                    "productive_eligible_TARGET_VERIFY_decisions": len(productive),
                    "productive_raw_bs_histogram": dict(collections.Counter(row["raw_bs"] for row in productive)),
                    "ineligible_TARGET_VERIFY_decisions": sum(row["mode"] == "TARGET_VERIFY" and row["eligible"] is False for row in decisions),
                    "all_new_modes_histogram": dict(collections.Counter(row["mode"] for row in decisions)),
                    "scope": "Functional capture to formal post snapshot, including serial prefix warmup; the complete original formal D metric archive independently proves within-formal Graph progression.",
                }
            native_graph = {}
            for line in obj["/metrics"]["body"].splitlines():
                if not line.startswith("sglang:cuda_graph_passes_total{"):
                    continue
                labels = dict(re.findall(r'(\w+)="([^"]*)"', line))
                rank = int(labels["moe_ep_rank"])
                assert labels["mode"] == "decode_cuda_graph" and labels["engine_type"] == "decode"
                assert int(labels["tp_rank"]) == rank and int(labels["dp_rank"]) == rank // 2
                assert rank not in native_graph
                native_graph[rank] = float(line.rsplit(" ", 1)[1])
            assert set(native_graph) == (set(range(16)) if node == 36 else set(range(16, 32)))
            assert all(value > 0 for value in native_graph.values())
        else:
            assert not aligned
        env = obj["actual_model_identity"]["selected_environment"]
        assert env["DEEP_USE_MODE"] == ("ops" if node in contract.D_NODES else "alltoall")
        assert env["HCCL_BUFFSIZE"] == ("256" if node in contract.D_NODES else "")
        assert not env["GLM53_PACKED_DISPATCH"]
        layouts = [row for row in events if row["event"] == "model_state_registered"]
        assert len(layouts) == 16
        for item in layouts:
            meta = item["metadata"]
            assert meta["hidden_kv_layers"] == 78 and meta["draft_kv_layers"] == 1
            assert meta["kv_buf_groups"] == 3 and len(meta["kv_item_lens"]) == 237
            assert len(meta["aux_item_lens"]) == 11 and meta["page_size"] == 128
            assert meta["kv_cache_dtype_str"] == "bfloat16"
        old_set = {json.dumps(row, sort_keys=True) for row in old}
        new = [row for row in events if json.dumps(row, sort_keys=True) not in old_set]
        assert not any(row["event"] in ["write_error", "model_state_send_error", "forbidden_tcp_model_payload"] for row in new)
        node_writes = [row for row in new if row["event"] == "write_complete"]
        for row in node_writes:
            assert row["rc"] == 0 and row["payload_bytes_completed"] == row["payload_bytes_requested"] > 0
            assert row["pid"] in init and node in contract.P_NODES
            assert row["context"]["kind"] in ["send_kvcache", "send_aux"]
            writes.append(row | {"node": node})
        assert all(row["rc"] == 0 for row in new if row["event"] == "model_state_send_complete")
        current_errors = [line for log in obj["logs"].values() for line in log["error_lines"]]
        old_errors = [line for log in previous["logs"].values() for line in log["error_lines"]]
        assert current_errors == old_errors, "New native errors require review"
        assert sum(log["error_line_count"] for log in obj["logs"].values()) == sum(log["error_line_count"] for log in previous["logs"].values())
        errors[str(node)] = {"preserved_existing_error_lines": len(current_errors), "new_error_lines": 0}
        if node in contract.P_NODES:
            delta = sum(log["native_sdma"]["copy_descriptor_bytes"] for log in obj["logs"].values()) - sum(
                log["native_sdma"]["copy_descriptor_bytes"] for log in previous["logs"].values()
            )
            assert delta == sum(row["payload_bytes_completed"] for row in node_writes)
            native_bytes += delta
            before = sdk_counts(previous, initial := set(init))
            after = sdk_counts(obj, initial)
            for pid, tags in after.items():
                if "TP_MMC_PY_BATCH_GET" not in tags:
                    continue
                deltas = {
                    name: {
                        key: entry[key] - before.get(pid, {}).get(name, {}).get(key, 0)
                        for key in ["begin", "good", "bad", "raw_TOTAL_column_sum"]
                    }
                    for name, entry in tags.items()
                }
                assert all(value["bad"] == 0 and value["begin"] == value["good"] for value in deltas.values())
                if deltas["TP_MMC_PY_BATCH_GET"]["good"] > 0:
                    read_workers.append({
                        "node": node,
                        "pid": pid,
                        "device_id": init[pid]["device_id"],
                        "counter_deltas": {
                            name: {
                                key: (str(value) if isinstance(value, Decimal) else value)
                                for key, value in entry.items()
                            }
                            for name, entry in deltas.items()
                        },
                    })
    assert {(row["node"], row["device_id"]) for row in read_workers} == {
        (node, device) for node in contract.P_NODES for device in range(16)
    }, "All32 P workers must read actual MemCache payload"
    assert sum(row["correct"] for row in mtp) > 0
    assert writes and read_workers and native_bytes == sum(row["payload_bytes_completed"] for row in writes)
    actual_d_devices = set()
    for row in writes:
        assert row["remote_session"] in sessions
        node, device = sessions[row["remote_session"]]
        assert node in contract.D_NODES
        actual_d_devices.add((node, device))
    assert actual_d_devices == {(node, device) for node in contract.D_NODES for device in range(16)}
    assert {row["context"]["kind"] for row in writes} == {"send_kvcache", "send_aux"}
    gate = objects[34]["actual_formal_cache_gate"]
    wire = read(f"runs/{run_id}/cache-gate-native.json")
    payload = json.loads(wire["stdout"])
    assert wire["rc"] == payload["query_rc"] == 0
    assert gate["report"] == report["cache_gate"]
    assert gate["sha256"] == payload["report_sha256"]
    cpu_path = contract.INDEPENDENT_METRICS_NODE
    cpu_wire = read(cpu_path)
    assert cpu_wire["rc"] == 0
    cpu = json.loads(cpu_wire["stdout"])
    sources[cpu_path] = sha(cpu_path)
    metrics = cpu["independent_formal_metric_archive_audit"]
    assert metrics["all_native_bodies_SHA_and_DP_gauges_recomputed"]
    assert not metrics["errors"] and metrics["sample_pairs"] == metrics["complete_pairs"]
    assert metrics["archive_sha256"] == client["D_metric_archive_sha256"]
    assert cpu["formal_report_identity"]["sha256"] == client["raw_formal_report_sha256"]
    assert set(productive_graph) == {str(index) for index in range(32)}
    graph = cpu["formal_native_target_graph_counters"]
    assert set(graph["delta_by_global_EP_rank"]) == {str(index) for index in range(32)}
    assert all(value > 0 for value in graph["delta_by_global_EP_rank"].values())
    assert graph["native_endpoint_rows"] == metrics["native_HTTP_rows"]
    native = {
        "project_id": "glm53",
        "framework": "sglang",
        "concurrency": concurrency,
        "run_id": run_id,
        "pass": True,
        "runtime_CIDs": runtime,
        "native_PD_write_batches": len(writes),
        "native_PD_descriptor_bytes": native_bytes,
        "PD_descriptor_bytes_by_state": {
            kind: sum(row["payload_bytes_completed"] for row in writes if row["context"]["kind"] == kind)
            for kind in ["send_kvcache", "send_aux"]
        },
        "native_D_devices_receiving_model_state": len(actual_d_devices),
        "physical_wire_bytes": None,
        "native_PD_request_ID_join": None,
        "native_PD_request_ID_join_limitation": "Existing SDK send audit exposes peer sessions and typed state, with room=null; period transfers/all32Ddevices are verified. Logical request success/count/cache/terminal joins are independently verified in AIS/client records. No invented per-request native transfer mapping.",
        "native_MemCache_read_workers": read_workers,
        "formal_native_cached_token_tiers": {key: dict(value) for key, value in tiers.items()},
        "storage_field_null_requests": null_counts,
        "storage_sum_scope": "Only observed numeric storage values summed; actual omitted fields are counted in storage_field_null_requests and remain null.",
        "storage_backend_annotation_counts": {key: dict(value) for key, value in storage_backends.items()},
        "native_cached_token_tiers_scope": "Native API accounting; D HiCache disabled. D storage label does not establish a direct D MemCache read; native SDK/PD evidence reported separately.",
        "formal_MTP": {
            "requests": len(mtp),
            "verify_steps": sum(row["verify"] for row in mtp),
            "proposed_drafts": sum(row["proposed"] for row in mtp),
            "correct_drafts": sum(row["correct"] for row in mtp),
            "num_drafts_per_verify": 5,
        },
        "retained_profile_file_metadata_unchanged": True,
        "native_P_mapping_reuse_and_packed_MTP_qualification": contract.READINESS_RESULT,
        "execution_mode": "Native P mapped-host AscendC IO; D NPU target full Graph and Ops fullmesh_v1; draft eager. All32 perrank Graph counters advanced within original formal metric archive; actual buckets unavailable",
        "formal_native_target_graph_counters": graph,
        "new_productive_graph_decisions_by_global_EP_rank": productive_graph,
        "non_SDK_installation_only_pids": installation_only_pids,
        "generic_execute_hook_replay_rows": 0,
        "generic_execute_hook_limitation": "NPU override bypasses generic execute; native counters and source execute replay path provide actual Graph evidence",
        "independent_complete_D_metrics_archive": metrics,
        "preserved_connection_diagnostics": errors,
        "D_eager_input_bs2_workers": 32,
        "D_eager_input_scope": "Functional all32 actual bs2 draft-DECODE rows are a prerequisite; formal capture is not used to invent a second bs2 proof.",
        "storage_annotation_helper": {
            "path": contract.CACHE_ANNOTATION_HELPER,
            "sha256": contract.CACHE_ANNOTATION_HELPER_SHA,
            "raw_D_storage_null_requests": reviewed["D_storage_null_requests"],
            "raw_nulls_filled": False,
        },
        "profiling": False,
        "retries": 0,
        "source_evidence_sha256": sources,
        "cache_annotation_review_sha256": sha(contract.FORMAL_CACHE_REVIEW),
        "original_client_baseline_accepted": report["baseline_accepted"],
    }
    native_path = contract.FORMAL_NATIVE_AUDIT
    (ROOT / native_path).write_text(json.dumps(native, indent=2) + "\n")
    certificate = client | {
        "measurement_valid": True,
        "native_postrun_audit_required": False,
        "native_postrun_evidence": native_path,
        "native_postrun_evidence_sha256": sha(native_path),
        "shared_run_id": settings["shared_run_id"],
        "runtime_CIDs": runtime,
        "accepted_baseline": False,
        "accepted_optimization_trial": False,
        "accepted_C32_restored_L04_control": False,
        "accepted_C32_target_graph_candidate_measurement": True,
        "formal_measurement_accepted_SLA_separate": True,
        "acceptance_basis": "Independent source-qualified D optional-field review, complete original client timing/count/input/gate checks and postrun native verification. Original raw client acceptance flag and optional-field nulls retained unchanged.",
        "sampled_peak_D_running": metrics["sampled_peak_total_D_running"],
        "native_proofs": {
            "actual_allprefix_payload_samples": True,
            "actual_PD_SDMA_period_state_and_all_D_devices": True,
            "actual_native_MemCache_reads": True,
            "actual_MTP5": True,
            "complete_D16_native_metrics": True,
            "actual_D_bs2_draft_DECODE_workers": 32,
            "native_execution_mode": "target_full_graph_draft_eager_Ops_fullmesh_v1",
        },
    }
    measurement_path = contract.FORMAL_MEASUREMENT
    (ROOT / measurement_path).write_text(json.dumps(certificate, indent=2) + "\n")
    print(json.dumps({
        "certificate": measurement_path,
        "sha256": sha(measurement_path),
        "measurement_valid": True,
        "aggregate_SLA_pass": certificate["aggregate_SLA_pass"],
        "output_tokens_per_second": certificate["output_tokens_per_second"],
    }))
    return certificate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", nargs="?", default=contract.FORMAL_RUN)
    parser.add_argument("capture_id", nargs="?", default=contract.FORMAL_CAPTURE)
    parser.add_argument("before_id", nargs="?", default=contract.FORMAL_BASELINE_CAPTURE)
    args = parser.parse_args()
    audit(args.run_id, args.capture_id, args.before_id)
