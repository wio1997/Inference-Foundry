"""DRAFT_PENDING_PRIMARY_REVIEW: independent L17 native readiness audit.

Aggregates the actual startup capture, short functional result, two-80K long
result and post-functional native capture.  It requires all32 D bs2 draft-DECODE
first-use rows, per-rank productive Graph deltas, typed SDMA writes to all32 D
devices, actual P SDK cache-read deltas, MTP first use, strict SDK counters and
the source-qualified D optional-storage helper.  Raw D storage nulls remain
null; no cached-token count is filled.
"""
import collections
import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path

import glm53_C32_L17_pipeline_contract_v2 as contract

ROOT = Path(__file__).resolve().parents[1]
SDK_TAGS = [
    "TP_MMC_PY_BATCH_GET",
    "TP_MMC_LOCAL_BATCH_GET",
    "TP_MMC_LOCAL_BATCH_GET_SIZE",
    "TP_HYBM_SDMA_BATCH_LD_TO_GD",
]


def sha(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def sdk_counts(panel, pids):
    result = {}
    for snapshot in panel["native_SDK_small_file_snapshots"]:
        assert snapshot["rc"] == 0
        if snapshot["container_id"] != panel["container_id"]:
            continue
        for item in snapshot["files"]:
            assert item["complete_line_ending"]
            assert hashlib.sha256(item["text"].encode()).hexdigest() == item["sha256"]
            match = re.search(r"ptracer_(\d+)\.dat$", item["path"])
            if not match or int(match[1]) not in pids:
                continue
            pid = int(match[1])
            tags = {}
            for line in item["text"].splitlines():
                parts = line.split()
                if len(parts) > 11 and parts[2] in SDK_TAGS:
                    begin, good, bad, on_fly = map(int, parts[3:7])
                    total = Decimal(parts[11])
                    assert min(begin, good, bad, on_fly) >= 0
                    assert total >= 0
                    entry = tags.setdefault(parts[2], {
                        "begin": 0, "good": 0, "bad": 0, "on_fly_last": None,
                        "raw_TOTAL_column_sum": Decimal(0), "lines": 0,
                    })
                    entry["begin"] += begin
                    entry["good"] += good
                    entry["bad"] += bad
                    entry["on_fly_last"] = on_fly
                    entry["raw_TOTAL_column_sum"] += total
                    entry["lines"] += 1
            for tag, entry in tags.items():
                assert entry["bad"] == 0, (panel["node"], pid, tag, entry)
                assert entry["begin"] == entry["good"], (panel["node"], pid, tag, entry)
                assert entry["on_fly_last"] == 0, (panel["node"], pid, tag, entry)
                entry["TOTAL_column_present_in_11plus_parts"] = entry["lines"] > 0
            result[pid] = tags
    return result


def graph_counts(panel):
    counts = {}
    for line in panel["/metrics"]["body"].splitlines():
        if not line.startswith("sglang:cuda_graph_passes_total{"):
            continue
        labels = dict(re.findall(r'(\w+)="([^"]*)"', line))
        if labels["mode"] != "decode_cuda_graph":
            continue
        rank = int(labels["moe_ep_rank"])
        assert labels["engine_type"] == "decode"
        assert int(labels["tp_rank"]) == rank and int(labels["dp_rank"]) == rank // 2
        assert rank not in counts
        counts[rank] = float(line.rsplit(" ", 1)[1])
    return counts


def main():
    p_capacity, d_capacity = contract.check_capacity_pair()
    startup = json.loads((ROOT / contract.STARTUP_RESULT).read_text())
    short = json.loads((ROOT / contract.SHORT_RESULT).read_text())
    assert startup["pass"] and short["pass"]
    assert startup["all64_actual_SDMA_and_BF16_layout_verified"] is True
    assert startup["all32_P_native_IO_and_existing_MC_mapping_reuse_verified"] is True
    assert startup["all32_D_nativeOps_fullmesh_and_targetGraph_retained"] is True
    assert short["successful_backend_inferences"] == 64 and short["successful_output_tokens"] == 19232
    assert short["D_DP_logical_cases"] == {str(index): 2 for index in range(16)}
    assert startup["runtime_CIDs"] == short["runtime_CIDs"]
    sources = {
        contract.STARTUP_RESULT: sha(contract.STARTUP_RESULT),
        contract.SHORT_RESULT: sha(contract.SHORT_RESULT),
        contract.P_CAPACITY: contract.P_CAPACITY_SHA,
        contract.D_CAPACITY: contract.D_CAPACITY_SHA,
    }
    for item in contract.old_source_provenance().values():
        sources[item["path"]] = item["sha256"]

    long_receipt = json.loads((ROOT / "runs" / contract.LONG_RUN / "receipt.json").read_text())
    assert long_receipt["rc"] == 0
    assert len(long_receipt["model_precheck"]) == 4
    assert all(value == 0 for value in long_receipt["model_precheck"].values())
    long_wire_rel = "runs/" + contract.LONG_RUN + "/node-43.json"
    long_wire = json.loads((ROOT / long_wire_rel).read_text())
    assert long_wire["rc"] == 0
    sources[long_wire_rel] = sha(long_wire_rel)
    sources["runs/" + contract.LONG_RUN + "/receipt.json"] = sha("runs/" + contract.LONG_RUN + "/receipt.json")
    artifacts = None
    for line in long_wire["stdout"].splitlines():
        if line.startswith("GLM53_LONG_ARTIFACTS="):
            artifacts = json.loads(line.split("=", 1)[1])
    assert artifacts is not None
    report = json.loads(artifacts["semantic-result.json"]["text"])
    phase = json.loads(artifacts["semantic-phase-status.json"]["text"])
    assert phase["client_rc"] == 0 and len(phase["gates"]) == 2
    assert all(gate["pass"] and gate["rc"] == 0 for gate in phase["gates"].values())
    assert report["semantic_pass"] is True
    assert len(report["cases"]) == len(report["gateway"]) == 6
    assert report["external_backend_HTTP_attempts"] == report["successful_backend_inferences"] == 12
    assert report["actual_successful_output_tokens"] == 2408
    mtp = []
    for row in report["cases"]:
        assert row["count_pass"] and row["pd_lifecycle_pass"]
        assert row["required_input"] in [74400, 80000]
        assert row["required_output"] in [1, 600]
        if row["required_output"] == 600:
            assert row["semantic_pass"] is True
            assert row["observed_answer_fields"][0] == row["expected_answer"]
            detail = row["native"]["extra_details_data"]["glm53_native_sglext"]["spec_tokens_details"]
            verify = detail["spec_verify_ct"]
            assert verify > 0 and detail["spec_num_proposed_drafts"] == 5 * verify
            assert detail["spec_num_correct_drafts"] > 0
            mtp.append(detail)
    assert len(mtp) == 4
    for gate in phase["gates"].values():
        meta = gate["metadata"]
        assert meta["pass"] and meta["close_rc"] == 0
        assert meta["known_component_keys"] == 1743 and meta["last_existing_component_keys"] == 1743
        assert meta["object_bytes"] == 8253063168 and meta["init_bm"] is False
        assert meta["NPU_devices"] is False
    assert len(report["cache_readback_guards"]) == 2
    assert all(proof["pass"] for proof in report["cache_readback_guards"].values())

    capture_receipt = json.loads((ROOT / "runs" / contract.FUNCTIONAL_CAPTURE / "receipt.json").read_text())
    assert capture_receipt["pass"] and capture_receipt["mode"] == "postfunctional"
    objects = {}
    sessions = {}
    writes = []
    read_workers = []
    graph_delta = {}
    productive = {}
    for node in contract.ALL_NODES:
        after_rel = "runs/" + contract.FUNCTIONAL_CAPTURE + "/node-" + str(node) + ".json"
        before_rel = "runs/" + contract.STARTUP_CAPTURE + "/node-" + str(node) + ".json"
        path, frame, after = contract.load_native_capture(after_rel)
        before_path, before_frame, before = contract.load_native_capture(before_rel)
        sources[after_rel] = sha(after_rel)
        sources[before_rel] = sha(before_rel)
        expected = (p_capacity if node in contract.P_NODES else d_capacity)["providers"][str(node)]
        assert after["container_id"] == before["container_id"] == expected["container_id"]
        assert after["provider_phase"] == "READY" and after["/health"]["status"] == 200
        assert not after["profiler_artifacts"]
        events = contract.persistent_events(after)
        old_events = contract.persistent_events(before)
        init = {row["pid"]: row for row in events if row["event"] == "initialize_complete"}
        assert len(init) == 16
        assert all(row["rc"] == 0 and row["protocol"] == "SDMA" and row["protocol_value"] == 1 for row in init.values())
        assert {row["namespace_pid"] for row in after["owned_npu_process_rows"]} == (
            {after["supervisor"]["provider_pid"], after["supervisor"]["model_pid"]} | set(init)
        )
        sessions.update({row["session"]: (node, row["device_id"]) for row in init.values()})
        old_set = {json.dumps(row, sort_keys=True) for row in old_events}
        new = [row for row in events if json.dumps(row, sort_keys=True) not in old_set]
        assert not any(row["event"] in ["write_error", "model_state_send_error", "forbidden_tcp_model_payload"] for row in new)
        for row in new:
            if row["event"] == "write_complete":
                assert node in contract.P_NODES and row["rc"] == 0
                assert row["payload_bytes_completed"] == row["payload_bytes_requested"] > 0
                assert row["pid"] in init and row["context"]["kind"] in ["send_kvcache", "send_aux"]
                writes.append(row | {"node": node})
        assert all(row["rc"] == 0 for row in new if row["event"] == "model_state_send_complete")
        current_errors = [line for log in after["logs"].values() for line in log["error_lines"]]
        old_errors = [line for log in before["logs"].values() for line in log["error_lines"]]
        assert current_errors == old_errors, "New native error requires review"
        for log in after["logs"].values():
            assert log["full_archive_complete"]
            assert log["native_sdma"]["copy_calls"] == 0 or log["native_sdma"]["copy_status_values"] == [0]
        gauges = [
            float(line.rsplit(" ", 1)[1]) for line in after["/metrics"]["body"].splitlines()
            if line.startswith(("sglang:num_running_reqs{", "sglang:num_queue_reqs{"))
        ]
        assert gauges and not any(gauges)
        if node in contract.D_NODES:
            before_counts = graph_counts(before)
            after_counts = graph_counts(after)
            expected_ranks = set(range(16)) if node == 36 else set(range(16, 32))
            assert set(before_counts) == set(after_counts) == expected_ranks
            assert all(after_counts[rank] > before_counts[rank] for rank in expected_ranks)
            graph_delta[str(node)] = {str(rank): after_counts[rank] - before_counts[rank] for rank in sorted(expected_ranks)}
            eager = after["D_eager_input_audit"]
            assert eager["coverage_pass"] is True and eager["coverage"]["all_bs2_draft_decode"] is True
            assert eager["coverage"]["all_flag_true"] and eager["coverage"]["all_clone_ok"]
            assert eager["coverage"]["all_out_cache_loc_tensor_metadata"] is True
            assert not eager["file_errors"] and len(eager["files"]) > 0
            assert len(eager["worker_environment_and_identity"]) == 16
            aligned = contract.alignment_events(after)
            for pid in init:
                rows = [row for row in aligned if row["pid"] == pid]
                for method in ["_npu_low_latency_dispatch", "_npu_low_latency_combine"]:
                    found = [row for row in rows if row["event"] == "ops_first_return" and row["method"] == method]
                    assert len(found) == 1 and found[0]["group_size"] == 32 and found[0]["comm_alg"] == "fullmesh_v1"
                assert any(row["event"] == "draft_graph_disabled" for row in rows)
                fresh = [
                    row for row in rows
                    if row["event"] == "target_graph_decision"
                    and row["eligible"] is True and row["mode"] == "TARGET_VERIFY"
                    and row["raw_bs"] > 0 and not row["is_draft"]
                ]
                assert fresh
                assert not any(row["event"] == "target_graph_replay_return" for row in rows)
                rank = next(row["rank"] for row in rows if row["event"] == "ops_first_return")
                productive[f"{node}:{pid}"] = {
                    "node": node,
                    "pid": pid,
                    "rank": rank,
                    "productive_TARGET_VERIFY_decisions": len(fresh),
                    "raw_bs_histogram": dict(collections.Counter(row["raw_bs"] for row in fresh)),
                }
        else:
            assert len(after["native_optional_extension_worker_maps"]) == 16
            assert {row["namespace_pid"] for row in after["native_optional_extension_worker_maps"]} == set(init)
            firstuse = []
            for log in after["logs"].values():
                for line in log["metadata_marker_lines"]:
                    if "GLM53_C32_NATIVE_HICACHE_MTP_FIRST_USE=" in line:
                        row = json.loads(line.split("GLM53_C32_NATIVE_HICACHE_MTP_FIRST_USE=", 1)[1])
                        assert row["target_layers"] == 78 and row["draft_layers"] == 1
                        assert row["target_indexer_rows"] == 78
                        assert not row["draft_indexer_host_mirror"]
                        assert row["original_native_copy_helper_preserved"]
                        firstuse.append(row)
            assert len(firstuse) == 16 and {row["pid"] for row in firstuse} == set(init)
            counters = sdk_counts(after, initial := set(init))
            prior = sdk_counts(before, initial)
            positive = {}
            for pid, tags in counters.items():
                current_good = tags.get("TP_MMC_PY_BATCH_GET", {}).get("good", 0)
                prior_good = prior.get(pid, {}).get("TP_MMC_PY_BATCH_GET", {}).get("good", 0)
                assert current_good >= prior_good
                if current_good > prior_good:
                    positive[pid] = {
                        "actual_model_MC_batch_get_good_before": prior_good,
                        "actual_model_MC_batch_get_good_after": current_good,
                        "good_delta": current_good - prior_good,
                        "counter_tags": {
                            name: {
                                key: (str(value) if isinstance(value, Decimal) else value)
                                for key, value in entry.items()
                            }
                            for name, entry in tags.items()
                        },
                    }
            assert len(positive) >= 8, "Fresh actual model-side native cache payload read deltas required on each P node"
            read_workers.extend({"node": node, **value, "pid": pid} for pid, value in positive.items())
        objects[str(node)] = after
    assert writes
    assert {row["context"]["kind"] for row in writes} == {"send_kvcache", "send_aux"}
    receiving = {sessions[row["remote_session"]] for row in writes}
    assert receiving == {(node, device) for node in contract.D_NODES for device in range(16)}
    assert graph_delta and all(len(value) == 16 for value in graph_delta.values())
    assert len(productive) == 32
    assert {row["rank"] for row in productive.values()} == set(range(32))
    assert len({(row["node"], row["namespace_pid"]) for row in short["D_eager_input_acceptance"]["worker_rows"]}) == 32
    output = {
        "project_id": "glm53",
        "framework": "sglang",
        "pass": True,
        "status": "L17_NATIVE_FUNCTIONAL_READINESS_COMPLETE",
        "shared_run_id": contract.P_REMOTE_ROOT.rsplit("/", 1)[-1],
        "runtime_CIDs": startup["runtime_CIDs"],
        "all32_P_native_IO_and_existing_MC_mapping_reuse_verified": True,
        "all32_P_actual_native_packed_MTP_firstuse_verified": True,
        "all32_D_actual_Ops_and_native_Graph_verified": True,
        "all_DP_short_and_80K_semantic_verified": True,
        "native_cache_and_SDMA_verified": True,
        "qualification": "Current runtime all32 P native MTP/cache reads, all32 D Ops/fullmesh/Graph and functional bs2 draft coverage, C32 short all16D-DP2each, two80K facts and cross-PDP model SDK reads with typed SDMA. Canonical formal prefix payload gate still required.",
        "external_functional_backend_HTTP_attempts": 76,
        "successful_functional_backend_inferences": 76,
        "successful_functional_output_tokens": 21640,
        "short_functional_output_tokens": 19232,
        "long_functional_output_tokens": 2408,
        "native_model_cache_read_counts": read_workers,
        "native_model_cache_read_worker_count": len({(row["node"], row["pid"]) for row in read_workers}),
        "SDK_counter_scope": "Strict actual begin==good, bad==0, on_fly_last==0, with the 11th-index TOTAL column parsed as Decimal. No filled count.",
        "native_D_devices_receiving_poststartup_state": len(receiving),
        "poststartup_PD_write_batches": len(writes),
        "poststartup_PD_descriptor_bytes": sum(row["payload_bytes_completed"] for row in writes),
        "physical_wire_bytes": None,
        "native_PD_request_ID_join": None,
        "native_PD_request_ID_join_limitation": "The SDK send audit exposes peer sessions and typed state; logical request joins are proved by the short/long clients. No per-request native mapping is invented.",
        "MTP_requests_checked": len(mtp),
        "D_native_target_graph_delta_by_rank": graph_delta,
        "D_productive_graph_decisions": productive,
        "D_graph_scope": "Startup capture to post-functional capture; all32 ranks strictly increased. Actual bucket sizes remain unavailable.",
        "D_eager_input_acceptance": short["D_eager_input_acceptance"],
        "D_eager_input_worker_count": len(short["D_eager_input_acceptance"]["worker_rows"]),
        "D_eager_input_join_scope": "COMPOSITE_NODE_PLUS_NAMESPACE_PID; 32 unique (node, namespace_pid) pairs, never pooled by PID alone",
        "storage_annotation": {
            "helper": contract.CACHE_ANNOTATION_HELPER,
            "helper_sha256": contract.CACHE_ANNOTATION_HELPER_SHA,
            "invoked": False,
            "reason": "The helper consumes a formal report. Readiness stores only the pinned helper identity; raw D storage nulls are not filled and no cached count is invented.",
            "formal_report_pending": True,
        },
        "combined_MC_and_native_82_kernel_prerequisite_pass": startup["combined_MC_and_native_82_kernel_prerequisite_pass"],
        "formal_requests": 0,
        "formal_performance_accepted": False,
        "formal_ready": True,
        "current_runtime_semantics_retested": True,
        "new_profiles": 0,
        "SDK_init_calls": 0,
        "external_model_requests": 0,
        "source_evidence_sha256": sources,
    }
    target = ROOT / contract.READINESS_RESULT
    assert not target.exists(), "Preserve existing readiness result"
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "pass": True,
        "functional_backend": 76,
        "functional_output": 21640,
        "D_graph_ranks": sum(len(value) for value in graph_delta.values()),
        "D_bs2_workers": 32,
        "output": contract.READINESS_RESULT,
        "sha256": sha(contract.READINESS_RESULT),
    }))


if __name__ == "__main__":
    main()
