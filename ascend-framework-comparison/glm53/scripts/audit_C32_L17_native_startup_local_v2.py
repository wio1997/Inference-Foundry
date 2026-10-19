"""DRAFT_PENDING_PRIMARY_REVIEW: independent L17 startup native audit.

Reads only the new v3 startup capture and frozen capacity/source records.
Identity is read from node-N-raw-wire.json before GLM53_NATIVE_GZIP and
independently re-verified with the pinned codec; the decoded panel is never used
to synthesize an identity record.  It requires actual all64 typed SDMA, the
237+11 BF16 target-78/draft-1 layout, P buffer_only/9GB/mapping-reuse/private
extension, D Ops fullmesh/Graph and the D NoCopy/audit worker environment.

Every actual D SDK worker must independently join a schema/source-valid
first-use row whose native_no_copy_get and clone_distinct are both true.  The
worker /proc environ is the exec initial environment only: it is reported as-is
with its expected-match mask, never equated with the later live EnvBool state
and never used to fill a first-use row.  D bs2 draft-DECODE completeness remains
a functional-audit requirement and may still be pending at startup.  It writes
no pass if any actual artifact is missing.
"""
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import glm53_C32_L17_pipeline_contract_v2 as contract

ROOT = Path(__file__).resolve().parents[1]
FIRST_USE_SCHEMA = "glm53.L17.eager_input_audit.v2"
FIRST_USE_AUDIT_VERSION = "v2"
D_WORKER_ENV_EXPECTED = {
    "SGLANG_EAGER_INPUT_NO_COPY": "1",
    "GLM53_EAGER_INPUT_AUDIT": "1",
    "GLM53_EAGER_INPUT_AUDIT_ROOT": contract.EAGER_ROOT_IN_CONTAINER,
    "GLM53_NODE": None,
    "TASK_QUEUE_ENABLE": "0",
    "HCCL_BUFFSIZE": "256",
}


def sha(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def worker_environment_expectation(node):
    expected = dict(D_WORKER_ENV_EXPECTED)
    expected["GLM53_NODE"] = str(node)
    return expected


def _tensor_meta(row, name):
    value = (row.get("allowlisted") or {}).get(name)
    if not isinstance(value, dict) or value.get("tensor") is not True:
        return None
    shape = value.get("shape")
    stride = value.get("stride")
    dtype = value.get("dtype")
    device = value.get("device")
    if not (
        isinstance(shape, list)
        and isinstance(stride, list)
        and len(shape) == len(stride)
        and all(type(item) is int and item >= 0 for item in shape)
        and all(type(item) is int for item in stride)
    ):
        return None
    if not (isinstance(dtype, str) and dtype and isinstance(device, str) and device):
        return None
    return {"shape": shape, "dtype": dtype, "device": device, "stride": stride}


def _valid_first_use_row(row, node, pid):
    """Validate one actual native first-use row against the pinned v2 schema.

    A stored row is only getter evidence when the live getter returned true and
    the clone was distinct.  No field is defaulted or coerced.
    """
    assert isinstance(row, dict), "first-use row is not an object"
    assert row.get("schema") == FIRST_USE_SCHEMA, "first-use schema differs"
    assert row.get("audit_version") == FIRST_USE_AUDIT_VERSION, "first-use audit version differs"
    assert row.get("project_id") == contract.PROJECT, "first-use project differs"
    assert row.get("source_sha256") == contract.EAGER_SOURCE_SHA, "first-use source SHA differs"
    assert row.get("eager_source_sha256") == contract.EAGER_SOURCE_SHA, "first-use eager source SHA differs"
    assert row.get("environ_source_sha256") == contract.ENVIRON_SOURCE_SHA, "first-use environ source SHA differs"
    assert row.get("node") == node, "first-use node differs"
    assert row.get("pid") == pid, "first-use pid differs"
    assert row.get("native_no_copy_get") is True, "native first-use getter is not True"
    assert row.get("clone_distinct") is True, "native first-use clone_distinct is not True"
    assert type(row.get("is_draft_worker")) is bool, "first-use draft flag missing"
    assert type(row.get("batch_size")) is int and row["batch_size"] >= 1, "first-use batch size missing"
    assert isinstance(row.get("forward_mode"), str) and row["forward_mode"], "first-use forward mode missing"
    assert type(row.get("dataclass_field_count")) is int and row["dataclass_field_count"] > 0, "first-use field count missing"
    verdicts = row.get("field_verdicts")
    assert isinstance(verdicts, dict) and verdicts, "first-use field verdicts missing"
    assert all(type(value) is int and value >= 0 for value in verdicts.values()), "first-use verdict type differs"
    assert sum(verdicts.values()) == row["dataclass_field_count"], "first-use verdict sum differs"
    digest = row.get("field_names_sha256")
    assert isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), "first-use field names SHA differs"
    allow = row.get("allowlisted")
    assert isinstance(allow, dict) and set(allow) == set(contract.AUDIT_ALLOWLIST_FIELDS), "first-use allowlist differs"
    return True


def first_use_coverage(d_eager, node, sdk_pids):
    """Re-derive per-worker live getter/clone coverage from actual rows only.

    Progress markers and worker environments are deliberately not read: every
    result below comes from D_eager_input_audit.records, and each record must be
    schema/source-valid.  The reported coverage counters are cross-checked
    against the re-derived rows so a printed flag cannot substitute for rows.
    """
    assert isinstance(d_eager, dict) and d_eager.get("applicable") is True, "D eager audit not applicable"
    files = d_eager.get("files")
    records = d_eager.get("records")
    file_errors = d_eager.get("file_errors")
    assert isinstance(files, list) and files, "D eager audit files missing"
    assert isinstance(records, list) and records, "D eager audit records missing"
    assert file_errors == [], "D eager audit has row errors"
    file_rows = {}
    for item in files:
        assert isinstance(item, dict), "D eager file row is not an object"
        path = item.get("path")
        assert isinstance(path, str) and path, "D eager file path missing"
        assert item.get("complete_line_ending") is True, "D eager file lacks a complete line ending"
        assert type(item.get("rows")) is int and item["rows"] >= 0, "D eager file row count missing"
        assert type(item.get("bytes")) is int and item["bytes"] >= 0, "D eager file byte count missing"
        assert isinstance(item.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]), "D eager file SHA differs"
        assert path not in file_rows, "duplicate D eager file path"
        file_rows[path] = item["rows"]
    by_pid = defaultdict(list)
    for item in records:
        assert isinstance(item, dict), "D eager record is not an object"
        path = item.get("path")
        line = item.get("line")
        row = item.get("row")
        assert path in file_rows, "D eager record references an unknown file"
        assert type(line) is int and 1 <= line <= file_rows[path], "D eager record line is out of range"
        pid = row.get("pid") if isinstance(row, dict) else None
        assert pid in sdk_pids, "D eager record pid is not an actual SDK worker"
        _valid_first_use_row(row, node, pid)
        by_pid[pid].append(item)
    assert not (set(by_pid) - set(sdk_pids))
    flag_true = sorted(pid for pid in sdk_pids if by_pid.get(pid))
    clone_ok = sorted(pid for pid in sdk_pids if by_pid.get(pid))
    assert set(flag_true) == set(sdk_pids) == set(clone_ok), "not every actual SDK worker joins a live getter/clone row"
    coverage = d_eager.get("coverage")
    assert isinstance(coverage, dict), "D eager coverage block missing"
    assert coverage.get("all_16_workers") is True, "reported all_16_workers is not true"
    assert coverage.get("all_flag_true") is True, "reported all_flag_true is not true"
    assert coverage.get("all_clone_ok") is True, "reported all_clone_ok is not true"
    assert coverage.get("no_row_errors") is True, "reported no_row_errors is not true"
    assert coverage.get("workers") == len(sdk_pids) == 16, "reported worker count differs from actual workers"
    assert coverage.get("flag_true_workers") == len(flag_true) == 16, "reported flag_true_workers differs from rows"
    assert coverage.get("clone_ok_workers") == len(clone_ok) == 16, "reported clone_ok_workers differs from rows"
    bs2_rows = []
    for pid in sorted(sdk_pids):
        for item in by_pid[pid]:
            row = item["row"]
            if row.get("is_draft_worker") is True and row.get("forward_mode") == "DECODE" and row.get("batch_size") == 2:
                bs2_rows.append(item)
    bs2_pids = sorted({item["row"]["pid"] for item in bs2_rows})
    out_cache = [
        item for item in bs2_rows
        if _tensor_meta(item["row"], "out_cache_loc") is not None
    ]
    out_cache_pids = sorted({item["row"]["pid"] for item in out_cache})
    assert coverage.get("draft_decode_bs2_workers") == len(bs2_pids), "reported bs2 worker count differs from rows"
    assert coverage.get("bs2_decode_rows") == len(bs2_rows), "reported bs2 row count differs from rows"
    assert coverage.get("out_cache_loc_workers") == len(out_cache_pids), "reported out_cache_loc worker count differs"
    assert coverage.get("out_cache_loc_rows") == len(out_cache), "reported out_cache_loc row count differs"
    assert type(coverage.get("all_bs2_draft_decode")) is bool
    assert type(coverage.get("all_out_cache_loc_tensor_metadata")) is bool
    return {
        "evidence_source": "D_eager_input_audit.records only; progress markers and worker environments are not used",
        "files": len(files),
        "rows": len(records),
        "actual_sdk_workers": len(sdk_pids),
        "workers_with_native_no_copy_get": len(flag_true),
        "workers_with_clone_distinct": len(clone_ok),
        "all_flag_true": True,
        "all_clone_ok": True,
        "no_row_errors": True,
        "bs2_observed_workers": len(bs2_pids),
        "bs2_observed_rows": len(bs2_rows),
        "bs2_required_at_startup": False,
        "bs2_scope": "Startup observes only; the full functional native audit still requires complete all32 bs2 draft-DECODE rows.",
        "coverage_as_reported": dict(coverage),
        "per_worker": [
            {
                "namespace_pid": pid,
                "valid_rows": len(by_pid[pid]),
                "native_no_copy_get": all(item["row"].get("native_no_copy_get") is True for item in by_pid[pid]),
                "clone_distinct": all(item["row"].get("clone_distinct") is True for item in by_pid[pid]),
                "bs2_rows": sum(
                    1 for item in by_pid[pid]
                    if item["row"].get("is_draft_worker") is True
                    and item["row"].get("forward_mode") == "DECODE"
                    and item["row"].get("batch_size") == 2
                ),
            }
            for pid in sorted(sdk_pids)
        ],
    }


def worker_initial_environment(workers, node):
    """Report raw /proc exec-initial worker values and their expected-match mask.

    Missing keys stay missing and their mask stays false.  This block is never
    treated as live EnvBool state and never fills a first-use row.
    """
    assert isinstance(workers, list) and len(workers) == 16, "D worker environment rows must be 16"
    expected = worker_environment_expectation(node)
    rows = []
    true_count = 0
    false_count = 0
    for worker in workers:
        env = worker.get("selected_environment")
        assert isinstance(env, dict), "worker initial environment is not an object"
        reported = worker.get("initial_environment_expected_flags_match")
        assert isinstance(reported, dict) and set(reported) == set(expected), "worker initial mask keys differ"
        assert all(type(value) is bool for value in reported.values()), "worker initial mask is not boolean"
        actual = {key: env.get(key) == value for key, value in expected.items()}
        assert reported == actual, "worker initial mask differs from the raw /proc initial bytes"
        raw = {key: env[key] for key in expected if key in env}
        missing = sorted(key for key in expected if key not in env)
        true_count += sum(1 for value in actual.values() if value is True)
        false_count += sum(1 for value in actual.values() if value is False)
        rows.append({
            "namespace_pid": worker.get("namespace_pid"),
            "host_pid": worker.get("host_pid"),
            "device_id": worker.get("device_id"),
            "initial_environment_raw": raw,
            "missing_initial_keys": missing,
            "initial_environment_expected_flags_match": dict(reported),
            "environment_scope": "Raw /proc exec initial bytes; missing keys remain missing and are not live EnvBool state.",
        })
    return rows, {
        "true": true_count,
        "false": false_count,
        "mask_keys_per_worker": len(expected),
        "mask_missing_forced_to_true": False,
    }


def identity_assertion_record(raw_wire, expected_node, expected_cid, expected_role, frame, codec):
    """Read identity only from the raw wire, then independently verify the frame.

    The decoded node-N.json has its ``stdout`` replaced by the decompressed JSON
    document, so the pre-GLM53_NATIVE_GZIP identity line exists only in
    node-N-raw-wire.json.  Never synthesize this record from a decoded panel.
    """
    assert type(raw_wire.get("rc")) is int and raw_wire["rc"] == 0, "raw wire rc != 0"
    assert isinstance(raw_wire.get("stdout"), str), "raw wire stdout missing"
    assert isinstance(frame, dict) and isinstance(frame.get("wire_stdout_sha256"), str), "decoded frame missing"
    raw_stdout = raw_wire["stdout"].encode()
    raw_stdout_sha256 = hashlib.sha256(raw_stdout).hexdigest()
    assert raw_stdout_sha256 == frame["wire_stdout_sha256"], "raw wire stdout SHA differs from decoded frame"
    decoded = codec.decode(raw_wire)
    assert decoded.get("native_output_transport") == frame, "pinned codec frame differs from decoded panel frame"
    assert len(decoded["stdout"].encode()) == frame["raw_bytes"], "decoded raw byte count differs"
    assert hashlib.sha256(decoded["stdout"].encode()).hexdigest() == frame["raw_sha256"], "decoded raw SHA differs"
    records = []
    for line in raw_wire["stdout"].splitlines():
        if line.startswith("GLM53_NATIVE_GZIP="):
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict) and "container_id" in row and "staged_files" in row:
            records.append(row)
    assert len(records) == 1, "exactly one identity assertion record required on the raw wire"
    record = records[0]
    assert record.get("project_id") == contract.PROJECT, "identity project mismatch"
    assert record.get("node") == expected_node, "identity node mismatch"
    assert record.get("container_id") == expected_cid, "identity container_id mismatch"
    assert record.get("role") == expected_role, "identity role mismatch"
    evidence = {
        "source": "node-N-raw-wire.json stdout, line before GLM53_NATIVE_GZIP",
        "raw_wire_stdout_sha256": raw_stdout_sha256,
        "native_output_transport": frame,
        "decoded_frame_equals_panel": True,
        "identity_assertion_record_count": 1,
        "codec_path": contract.CODEC,
        "codec_sha256": contract.CODEC_SHA,
    }
    return record, evidence


def main():
    p_capacity, d_capacity = contract.check_capacity_pair()
    capture = ROOT / "runs" / contract.STARTUP_CAPTURE
    receipt = json.loads((capture / "receipt.json").read_text())
    assert receipt.get("pass") is True and receipt.get("mode") == "startup"
    assert len(receipt.get("nodes", {})) == 4
    assert all(type(value) is int and value == 0 for value in receipt["nodes"].values())
    sources = {
        "results/sglang-C32-L12C-selected-P-pool-capacity-v1.json": contract.P_CAPACITY_SHA,
        "results/sglang-C32-L17-selected-D-pool-capacity-v2.json": contract.D_CAPACITY_SHA,
    }
    for item in contract.old_source_provenance().values():
        sources[item["path"]] = item["sha256"]
    nodes = {}
    all_d_ranks = set()
    codec = contract.load_module("l17_startup_audit_codec", contract.CODEC, contract.CODEC_SHA)
    sources[contract.CODEC] = contract.CODEC_SHA
    for node in contract.ALL_NODES:
        relative = "runs/" + contract.STARTUP_CAPTURE + "/node-" + str(node) + ".json"
        raw_relative = "runs/" + contract.STARTUP_CAPTURE + "/node-" + str(node) + "-raw-wire.json"
        path, frame, panel = contract.load_native_capture(relative)
        raw_path = ROOT / raw_relative
        assert raw_path.is_file() and not raw_path.is_symlink(), "raw wire required for identity"
        raw_wire = json.loads(raw_path.read_text())
        sources[relative] = sha(relative)
        sources[raw_relative] = sha(raw_relative)
        expected = (p_capacity if node in contract.P_NODES else d_capacity)["providers"][str(node)]
        assert panel["container_id"] == expected["container_id"]
        assert panel["supervisor"]["phase"] == "MODEL_RUNNING"
        assert panel["supervisor"].get("model_started") is True and "model_exit_code" not in panel["supervisor"]
        assert panel["provider_phase"] == "READY"
        identity = panel["actual_model_identity"]
        assert identity["namespace_pid"] == panel["supervisor"]["model_pid"]
        identity_meta, identity_evidence = identity_assertion_record(
            raw_wire,
            node,
            expected["container_id"],
            "P" if node in contract.P_NODES else "D",
            frame,
            codec,
        )
        identity_evidence["raw_wire"] = raw_relative
        identity_evidence["raw_wire_sha256"] = sources[raw_relative]
        assert identity_meta["container_id"] == panel["container_id"]
        assert identity_meta["name"] == expected["name"]
        assert identity_meta["run_id"] == expected["run_id"]
        assert identity_meta["model_requests"] == 0
        assert identity_meta["staged_files"] > 0
        assert identity_meta["optional_extension_mounted"] is (node in contract.P_NODES)
        events = contract.persistent_events(panel)
        init = {row["pid"]: row for row in events if row["event"] == "initialize_complete"}
        layout = [row for row in events if row["event"] == "model_state_registered"]
        assert len(init) == len(layout) == 16
        assert all(row["rc"] == 0 and row["protocol"] == "SDMA" and row["protocol_value"] == 1 for row in init.values())
        for row in layout:
            meta = row["metadata"]
            assert meta["hidden_kv_layers"] == 78 and meta["draft_kv_layers"] == 1
            assert meta["kv_buf_groups"] == 3
            assert len(meta["kv_item_lens"]) == 237 and len(meta["aux_item_lens"]) == 11
            assert meta["page_size"] == 128 and meta["kv_cache_dtype_str"] == "bfloat16"
        owned = panel["owned_npu_process_rows"]
        assert len(owned) == 18
        owned_pids = {row["namespace_pid"] for row in owned}
        assert owned_pids == ({panel["supervisor"]["provider_pid"], panel["supervisor"]["model_pid"]} | set(init))
        assert panel["/health"]["status"] == panel["/metrics"]["status"] == 200
        if node in (34, 36):
            assert panel["/get_server_info"]["status"] == 200
        else:
            assert panel["/get_server_info"] == {"error": "<HTTPError 404: 'Not Found'>"}
        leader = node if node in (34, 36) else (34 if node == 35 else 36)
        leader_panel = contract.load_native_capture(
            "runs/" + contract.STARTUP_CAPTURE + "/node-" + str(leader) + ".json"
        )[2]
        assert leader_panel["/get_server_info"]["status"] == 200
        info = json.loads(leader_panel["/get_server_info"]["body"])
        assert info["tp_size"] == info["ep_size"] == 32
        assert info["dp_size"] == (4 if node in contract.P_NODES else 16)
        assert info["speculative_num_steps"] == 5 and info["speculative_num_draft_tokens"] == 6
        assert info["kv_cache_dtype"] == "bfloat16" and info["disable_overlap_schedule"]
        floor = contract.P_KV_FLOOR if node in contract.P_NODES else contract.D_KV_FLOOR
        assert info["max_total_num_tokens"] >= floor
        if node in contract.P_NODES:
            assert info["hicache_host_memory_mode"] == "buffer_only" and info["hicache_size"] == 9
        else:
            assert info["cuda_graph_config"]["decode"]["backend"] == "full"
            assert info["cuda_graph_config"]["decode"]["bs"] == [1, 2, 3, 4]
        metrics = panel["/metrics"]
        assert hashlib.sha256(metrics["body"].encode()).hexdigest() == metrics["sha256"]
        gauges = [
            float(line.rsplit(" ", 1)[1]) for line in metrics["body"].splitlines()
            if line.startswith(("sglang:num_running_reqs{", "sglang:num_queue_reqs{"))
        ]
        assert gauges and not any(gauges)
        logs = list(panel["logs"].values())
        errors = []
        markers = []
        for log in logs:
            assert log["full_archive_complete"]
            assert log["error_line_count"] == len(log["error_lines"])
            errors += log["error_lines"]
            markers += log["metadata_marker_lines"]
            assert log["native_sdma"]["copy_calls"] == 0 or log["native_sdma"]["copy_status_values"] == [0]
        benign = r"^\d{4}-\d{2}-\d{2} [0-9:.]+ ERROR \[\d+-\d+\]\[AccLink acc_tcp_port_util\.cpp:(?:43|62) BindTcpPortV[46]\] bind ipv[46] tcp port failed, port=\d+, errno=98, errstr=Address already in use$"
        assert all(re.fullmatch(benign, line) for line in errors), errors
        assert not any("FATAL" in line or "GLM53_C32_PACKED_DISPATCH_" in line for line in markers)
        selected = identity["selected_environment"]
        argv = identity["actual_model_argv"]
        assert selected["ASCEND_MF_TRANSFER_PROTOCOL"] == "sdma"
        assert selected["SGLANG_MOONCAKE_SEND_AUX_TCP"] == "0"
        assert selected["PYTHONHASHSEED"] == "0"
        assert selected["ASCEND_ENABLE_USE_FABRIC_MEM"] == "1"
        assert selected["ACL_OP_INIT_MODE"] == "1"
        assert selected["GLM53_PACKED_DISPATCH"] == ""
        assert selected["GLM53_BUFFER_DEDUP_VALIDATE"] == ""
        assert selected["MEMFABRIC_HYBRID_EXTEND_LIB_PATH"] == ""
        assert selected["PYTHONPATH"].startswith("/opt/glm53/source/python:")
        assert selected["MMC_LOCAL_CONFIG_PATH"] == "/opt/glm53/source/config/local.conf"
        assert "--disable-overlap-schedule" in argv and "--disable-prefill-cuda-graph" in argv
        assert argv[argv.index("--model-path") + 1] == "/data/wio/models/GLM-5.3-w8a8"
        assert argv[argv.index("--quantization") + 1] == "modelslim"
        assert argv[argv.index("--dtype") + 1] == argv[argv.index("--kv-cache-dtype") + 1] == "bfloat16"
        assert argv[argv.index("--tp-size") + 1] == argv[argv.index("--ep-size") + 1] == "32"
        assert argv[argv.index("--speculative-num-steps") + 1] == "5"
        assert argv[argv.index("--speculative-num-draft-tokens") + 1] == "6"
        d_eager = None
        if node in contract.P_NODES:
            assert selected["DEEP_USE_MODE"] == "alltoall"
            assert selected["HCCL_BUFFSIZE"] == "" and selected["TASK_QUEUE_ENABLE"] == ""
            assert selected["GLM53_C32_TARGET_GRAPH_ALIGN"] == ""
            assert selected["GLM53_C32_NATIVE_HICACHE_MTP"] == "1"
            assert selected["GLM53_C32_NATIVE_MC_MAPPING_REUSE"] == "1"
            assert selected["SGLANG_HICACHE_HOST_MEM_BACKEND"] == "memfabric"
            assert argv[argv.index("--dp-size") + 1] == "4"
            assert argv[argv.index("--max-running-requests") + 1] == "8"
            assert "--disable-decode-cuda-graph" in argv
            assert argv[argv.index("--deepep-mode") + 1] == "normal"
            assert len(panel["native_optional_extension_worker_maps"]) == 16
            assert {x["namespace_pid"] for x in panel["native_optional_extension_worker_maps"]} == set(init)
            reused = {}
            installed = {}
            for line in markers:
                if "GLM53_C32_NATIVE_MC_MAPPING_REUSED=" in line:
                    row = json.loads(line.split("GLM53_C32_NATIVE_MC_MAPPING_REUSED=", 1)[1])
                    assert row["pid"] in init and row["existing_linear_mapping"]
                    assert not row["explicit_second_MC_registration"]
                    assert row["component"] not in reused.setdefault(row["pid"], {})
                    reused[row["pid"]][row["component"]] = row
                if "GLM53_C32_NATIVE_HICACHE_INSTALLED=" in line:
                    row = json.loads(line.split("GLM53_C32_NATIVE_HICACHE_INSTALLED=", 1)[1])
                    installed.setdefault(row["pid"], set()).add(row["module"])
            assert set(reused) == set(init)
            for pid in init:
                assert set(reused[pid]) == {"k_buffer", "v_buffer", "index_k_buffer"}
                assert installed[pid] == {
                    "sglang.srt.mem_cache.pool_host.mla",
                    "sglang.srt.mem_cache.storage.npu_memcache.npu_memcache_store",
                }
                for name, shape, size in [
                    ("k_buffer", [634, 79, 128, 1, 512], 6564872192),
                    ("v_buffer", [634, 79, 128, 1, 64], 820609024),
                    ("index_k_buffer", [634, 78, 128, 1, 128], 1620443136),
                ]:
                    assert reused[pid][name]["shape"] == shape and reused[pid][name]["bytes"] == size
        else:
            assert selected["DEEP_USE_MODE"] == "ops"
            assert selected["HCCL_BUFFSIZE"] == "256" and selected["TASK_QUEUE_ENABLE"] == "0"
            assert selected["GLM53_C32_TARGET_GRAPH_ALIGN"] == "1"
            assert selected["GLM53_NODE"] == str(node)
            assert selected["SGLANG_EAGER_INPUT_NO_COPY"] == "1"
            assert selected["GLM53_EAGER_INPUT_AUDIT"] == "1"
            assert selected["GLM53_EAGER_INPUT_AUDIT_ROOT"] == contract.EAGER_ROOT_IN_CONTAINER
            assert selected["SGLANG_HICACHE_HOST_MEM_BACKEND"] == ""
            assert selected["GLM53_C32_NATIVE_HICACHE_MTP"] == ""
            assert selected["GLM53_C32_NATIVE_MC_MAPPING_REUSE"] == ""
            assert argv[argv.index("--dp-size") + 1] == "16"
            assert argv[argv.index("--max-running-requests") + 1] == "64"
            assert argv[argv.index("--deepep-mode") + 1] == "low_latency"
            assert argv[argv.index("--cuda-graph-backend-decode") + 1] == "full"
            assert argv[argv.index("--cuda-graph-max-bs-decode") + 1] == "4"
            assert "--disable-decode-cuda-graph" not in argv
            aligned = contract.alignment_events(panel)
            for pid in init:
                records = [row for row in aligned if row["pid"] == pid]
                for method in ["_npu_low_latency_dispatch", "_npu_low_latency_combine"]:
                    found = [
                        row for row in records
                        if row["event"] == "ops_first_return" and row["method"] == method
                    ]
                    assert len(found) == 1 and found[0]["group_size"] == 32
                    assert found[0]["comm_alg"] == "fullmesh_v1"
                    all_d_ranks.add(found[0]["rank"])
                assert any(row["event"] == "draft_graph_disabled" for row in records)
            counts = {}
            for line in metrics["body"].splitlines():
                if line.startswith("sglang:cuda_graph_passes_total{"):
                    labels = dict(re.findall(r'(\w+)="([^"]*)"', line))
                    if labels["mode"] == "decode_cuda_graph":
                        counts[int(labels["moe_ep_rank"])] = float(line.rsplit(" ", 1)[1])
            expected_ranks = set(range(16)) if node == 36 else set(range(16, 32))
            assert set(counts) == expected_ranks and all(value > 0 for value in counts.values())
            d_eager = panel.get("D_eager_input_audit")
            assert isinstance(d_eager, dict) and d_eager.get("applicable") is True
            workers = d_eager["worker_environment_and_identity"]
            assert len(workers) == 16
            assert {row["namespace_pid"] for row in workers} == set(init)
            assert len({row["host_pid"] for row in workers}) == 16
            assert sorted(row["device_id"] for row in workers) == list(range(16))
            assert all(row["alive"] and row["cgroup_contains_current_cid"] for row in workers)
            assert all(row["other_environment_retained"] is False for row in workers)
            assert all(row["full_environment_hashed"] is False for row in workers)
            worker_env, worker_env_mask = worker_initial_environment(workers, node)
            first_use = first_use_coverage(d_eager, node, set(init))
        nodes[str(node)] = {
            "container_id": panel["container_id"],
            "actual_SDMA_workers": len(init),
            "state_workers": len(layout),
            "KV_capacity": info["max_total_num_tokens"],
            "KV_capacity_floor": floor,
            "KV_capacity_source_node": leader,
            "KV_capacity_scope": "Role-global native primary with distributed worker registrations; secondary API 404 preserved",
            "native_extension_worker_maps": len(panel.get("native_optional_extension_worker_maps", [])),
            "preserved_bind_error_lines": len(errors),
            "healthy_idle": True,
            "D_eager_input_audit": None if d_eager is None else {
                "coverage_status": d_eager["coverage_status"],
                "coverage_pass_as_reported": d_eager["coverage_pass"],
                "coverage": d_eager["coverage"],
                "files": len(d_eager["files"]),
                "workers": len(workers),
                "first_use": first_use,
                "worker_initial_environment": worker_env,
                "worker_initial_environment_mask_counts": worker_env_mask,
                "worker_initial_environment_scope": "Raw /proc exec initial bytes only; the expected-match mask preserves missing keys as false. This is not live EnvBool state; live evidence is the per-worker first-use rows.",
            },
            "native_output_transport": frame,
            "identity_render_assertion_record": identity_meta,
            "identity_render_assertion_evidence": identity_evidence,
        }
    assert all_d_ranks == set(range(32))
    assert all(row["D_eager_input_audit"] is not None for key, row in nodes.items() if key in ("36", "41"))
    output = {
        "project_id": "glm53",
        "framework": "sglang",
        "pass": True,
        "status": "NATIVE_STARTUP_CAPTURE_AUDIT_COMPLETE",
        "shared_run_id": contract.P_REMOTE_ROOT.rsplit("/", 1)[-1],
        "runtime_CIDs": {str(node): nodes[str(node)]["container_id"] for node in contract.ALL_NODES},
        "nodes": nodes,
        "all64_actual_SDMA_and_BF16_layout_verified": True,
        "all64_actual_source_weights_image_CLI_and_role_specific_execution_settings_verified": True,
        "role_specific_execution_settings_scope": "D only: parent identity TASK_QUEUE_ENABLE=0 and HCCL_BUFFSIZE=256. P retains its original empty TASK_QUEUE_ENABLE/HCCL_BUFFSIZE values. Worker /proc exec-initial bytes are reported raw and are not equated with live state.",
        "source_weights_image_scope": "Approved identity host code verifies actual container image ID, staged source SHA set and P private-extension mount; this audit rechecks actual container IDs, model argv, selected environment and the identity render assertion record. No extra remote call.",
        "all32_P_native_IO_and_existing_MC_mapping_reuse_verified": True,
        "all32_D_nativeOps_fullmesh_and_targetGraph_retained": True,
        "P_KV_capacity_floor": contract.P_KV_FLOOR,
        "D_KV_capacity_floor": contract.D_KV_FLOOR,
        "old_D_reported_KV_capacity": contract.OLD_D_KV,
        "native_layout": "237 kv_item_lens + 11 aux_item_lens, BF16 target78/draft1, MTP5/6",
        "D_eager_input_startup_scope": "All 16 actual SDK workers per D node must join schema/source-valid first-use rows with native_no_copy_get=True and clone_distinct=True. Worker /proc initial flags are reported raw with the expected-match mask and are never treated as live. bs2 draft-DECODE completeness may remain pending here; it is required by the functional native audit.",
        "D_graph_counter_scope": "Startup warmup counters only; productive per-rank deltas are mandatory in the functional audit.",
        "combined_MC_and_native_82_kernel_prerequisite_pass": True,
        "native_P_model_cache_semantics": "PENDING_SHORT_AND_TWO_80K_CACHE_SDMA",
        "storage_annotation": {
            "helper": contract.CACHE_ANNOTATION_HELPER,
            "helper_sha256": contract.CACHE_ANNOTATION_HELPER_SHA,
            "applies_to": "formal report rows only; raw D storage nulls remain null and are not filled here",
            "invoked": False,
        },
        "external_model_requests": 0,
        "SDK_init_calls": 0,
        "new_profiles": 0,
        "formal_qualified": False,
        "source_evidence_sha256": sources,
        "capture_id": contract.STARTUP_CAPTURE,
    }
    target = ROOT / contract.STARTUP_RESULT
    assert not target.exists(), "Preserve existing startup result"
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "pass": True,
        "actual_SDMA_workers": 64,
        "P_native_maps": 32,
        "D_capacity_floor": contract.D_KV_FLOOR,
        "old_D_reported_capacity": contract.OLD_D_KV,
        "output": contract.STARTUP_RESULT,
        "sha256": sha(contract.STARTUP_RESULT),
    }))


if __name__ == "__main__":
    main()
