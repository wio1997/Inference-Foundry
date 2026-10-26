"""DRAFT_PENDING_PRIMARY_REVIEW: independent L17 short functional/D eager audit v2.

Uses the actual short cohort and the post-functional native capture only.  It
requires exactly 32 logical / 64 backend requests, two per D DP, 19232 output
tokens, all32 actual draft-DECODE bs2 first-use rows with real out_cache_loc
metadata, and per-rank productive Graph deltas during the functional window.

v2 corrects the native evidence split already used by startup audit v2: the
worker /proc exec-initial environment is raw evidence with an expected-match
mask, and the six expected keys may legitimately be absent.  A missing initial
key is never filled, never treated as live EnvBool state, and never used as
first-use proof.  Live coverage comes only from actual D_eager_input_audit
records, each schema/source-valid with native_no_copy_get and clone_distinct
true.  The actual model parent process environment is checked separately.

The non-SDK alignment rows are import-time installation metadata.  Their
before/after rows must be identical and contain only ops_compat_installed and
target_graph_audit_installed; they are never counted as worker or Graph proof.
It never fills a missing row, tensor field or Graph delta.
"""
import collections
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import glm53_C32_L17_pipeline_contract_v2 as contract

ROOT = Path(__file__).resolve().parents[1]
STARTUP_HELPER = "scripts/audit_C32_L17_native_startup_local_v2.py"
STARTUP_HELPER_SHA = "715192aaff62c42cbcc0534e2ebd4319c7bd44f08dae71217af99e6886bdc597"
BEFORE_CAPTURE = "20261019T-L17-sglang-native-startup-capture-v3"
AFTER_CAPTURE = "20261019T-L17-sglang-native-functional-capture-v1"
NON_SDK_ALIGNMENT_EVENTS = frozenset((
    "ops_compat_installed",
    "target_graph_audit_installed",
))
PARENT_ENV_EXPECTED = {
    "SGLANG_EAGER_INPUT_NO_COPY": "1",
    "GLM53_EAGER_INPUT_AUDIT": "1",
    "GLM53_EAGER_INPUT_AUDIT_ROOT": contract.EAGER_ROOT_IN_CONTAINER,
    "GLM53_NODE": None,
    "TASK_QUEUE_ENABLE": "0",
    "HCCL_BUFFSIZE": "256",
}
BENIGN_BIND = re.compile(
    r"^\d{4}-\d{2}-\d{2} [0-9:.]+ ERROR \[\d+-\d+\]\[AccLink acc_tcp_port_util\.cpp:"
    r"(?:43|62) BindTcpPortV[46]\] bind ipv[46] tcp port failed, port=\d+, "
    r"errno=98, errstr=Address already in use$"
)


def sha(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def load_startup_helper():
    """SHA-pin the startup helper before import and never execute its main."""
    path = ROOT / STARTUP_HELPER
    assert path.is_file() and not path.is_symlink(), "startup helper is not a regular file"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == STARTUP_HELPER_SHA, "startup helper SHA differs: " + digest
    spec = importlib.util.spec_from_file_location(
        "glm53_l17_native_startup_helper_v2", path
    )
    assert spec is not None and spec.loader is not None, "startup helper spec missing"
    module = importlib.util.module_from_spec(spec)
    assert module.__name__ != "__main__", "startup helper must not run as main"
    spec.loader.exec_module(module)
    for name in ("first_use_coverage", "worker_initial_environment"):
        assert callable(getattr(module, name, None)), "startup helper function missing: " + name
    return module


STARTUP = load_startup_helper()


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
        assert int(labels["tp_rank"]) == rank
        assert int(labels["dp_rank"]) == rank // 2
        assert rank not in counts
        counts[rank] = float(line.rsplit(" ", 1)[1])
    return counts


def actual_parent_environment(panel, node):
    """Check the actual model parent environment, not the worker /proc bytes."""
    identity = panel.get("actual_model_identity")
    assert isinstance(identity, dict), "actual model identity missing"
    environment = identity.get("selected_environment")
    assert isinstance(environment, dict), "actual model parent environment missing"
    expected = dict(PARENT_ENV_EXPECTED)
    expected["GLM53_NODE"] = str(node)
    actual = {key: environment.get(key) for key in expected}
    assert actual == expected, "actual model parent environment differs: " + json.dumps(actual)
    return {
        "namespace_pid": identity.get("namespace_pid"),
        "expected": expected,
        "actual": actual,
        "scope": "Actual model parent process environment; separate from worker /proc exec-initial bytes.",
    }


def capture_lifecycle(panel, node, init_rows):
    """Preserve the SDMA/lifecycle gates on each immutable capture."""
    supervisor = panel.get("supervisor")
    assert isinstance(supervisor, dict), "supervisor block missing"
    assert supervisor.get("phase") == "MODEL_RUNNING", "model is not running"
    assert supervisor.get("model_started") is True, "model start not recorded"
    assert "model_exit_code" not in supervisor, "model exit recorded"
    assert panel.get("provider_phase") == "READY", "provider is not READY"
    identity = panel.get("actual_model_identity")
    assert isinstance(identity, dict), "actual model identity missing"
    assert identity.get("namespace_pid") == supervisor.get("model_pid"), "identity model pid differs"
    assert len(init_rows) == 16, "SDMA init rows must be 16"
    assert all(
        row.get("rc") == 0
        and row.get("protocol") == "SDMA"
        and row.get("protocol_value") == 1
        for row in init_rows
    ), "not all actual workers initialized over SDMA"
    owned = panel.get("owned_npu_process_rows")
    assert isinstance(owned, list) and len(owned) == 18, "owned NPU process rows must be 18"
    assert {row["namespace_pid"] for row in owned} == (
        {supervisor["provider_pid"], supervisor["model_pid"]}
        | {row["pid"] for row in init_rows}
    ), "owned NPU process rows differ from provider/model/SDMA workers"
    assert panel["/health"]["status"] == panel["/metrics"]["status"] == 200, "health/metrics not 200"
    metrics = panel["/metrics"]
    assert hashlib.sha256(metrics["body"].encode()).hexdigest() == metrics["sha256"], "metrics body SHA differs"
    gauges = [
        float(line.rsplit(" ", 1)[1]) for line in metrics["body"].splitlines()
        if line.startswith(("sglang:num_running_reqs{", "sglang:num_queue_reqs{"))
    ]
    assert gauges and not any(gauges), "capture is not idle"
    errors = []
    logs = list(panel["logs"].values())
    assert logs, "logs missing"
    for log in logs:
        assert log["full_archive_complete"], "log archive incomplete"
        assert log["error_line_count"] == len(log["error_lines"]), "log error count differs"
        assert log["native_sdma"]["copy_calls"] == 0 or log["native_sdma"]["copy_status_values"] == [0], "native SDMA copy gate failed"
        errors += log["error_lines"]
    assert all(BENIGN_BIND.fullmatch(line) for line in errors), errors
    return {
        "container_id": panel["container_id"],
        "node": node,
        "SDMA_workers": len(init_rows),
        "owned_NPU_process_rows": len(owned),
        "healthy_idle": True,
        "preserved_bind_error_lines": len(errors),
        "scope": "Actual initialized workers, owned processes, health/metrics and native SDMA log gate.",
    }


def non_sdk_alignment_evidence(before_rows, after_rows, sdk_pids):
    """Require non-SDK installation metadata to be unchanged installation only.

    These rows are emitted at import time and are not first-use or productive
    Graph evidence.  A new non-SDK execution row, an unknown event, or any
    metadata change fails the capture pair.
    """
    before_non = [row for row in before_rows if row.get("pid") not in sdk_pids]
    after_non = [row for row in after_rows if row.get("pid") not in sdk_pids]
    assert all(isinstance(row, dict) for row in before_non + after_non), "non-SDK alignment row is not an object"
    assert all("event" in row and "pid" in row for row in before_non + after_non), "non-SDK alignment row lacks event/pid"
    canonical = lambda rows: sorted(
        json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows
    )
    assert canonical(after_non) == canonical(before_non), "non-SDK alignment rows changed between captures"
    assert after_non, "non-SDK installation metadata missing"
    assert all(
        row["event"] in NON_SDK_ALIGNMENT_EVENTS for row in after_non
    ), "unexpected non-SDK alignment event"
    return {
        "rows": len(after_non),
        "pids": sorted({row["pid"] for row in after_non}),
        "event_counts": dict(collections.Counter(row["event"] for row in after_non)),
        "allowed_events": sorted(NON_SDK_ALIGNMENT_EVENTS),
        "before_equals_after": True,
        "productive_worker_proof": False,
        "scope": "Import-time installation metadata for non-SDK PIDs only; identical in startup-v3 and post-functional captures; never treated as worker first-use or Graph progress.",
    }


def main():
    p_capacity, d_capacity = contract.check_capacity_pair()
    assert contract.STARTUP_CAPTURE == BEFORE_CAPTURE, "startup capture is not the pinned v3 capture"
    assert contract.FUNCTIONAL_CAPTURE == AFTER_CAPTURE, "post-functional capture name differs"
    short_run = ROOT / "runs" / contract.SHORT_RUN
    receipt = json.loads((short_run / "receipt.json").read_text())
    assert receipt["rc"] == 0 and len(receipt["prechecks"]) == 4
    assert all(value == 0 for value in receipt["prechecks"].values())
    wire = json.loads((short_run / "node-34.json").read_text())
    assert wire["rc"] == 0
    frames = [
        line.split("=", 1)[1] for line in wire["stdout"].splitlines()
        if line.startswith("GLM53_SG_MATRIX_ARTIFACTS=")
    ]
    assert len(frames) == 1
    result = json.loads(frames[0])
    assert result["client_rc"] == 0
    report = result["result"]
    assert report["pass"] and report["offered_concurrency"] == 32
    assert len(report["cases"]) == 32
    assert len(report["backend_attempts"]) == report["model_requests_attempted"] == 64
    assert report["successful_inferences"] == 64
    assert report["actual_successful_output_tokens"] == report["successful_output_tokens"] == 19232
    assert report["input_tokens"] == 512 and report["input_sha256_le_u32"] == contract.INPUT512_SHA
    assert report["D_output_tokens"] == 600 and report["D_min_tokens"] == 600
    identifiers = set()
    dp = collections.Counter()
    for row in report["cases"]:
        assert row["pass"]
        prefill = row["prefill"]
        decode = row["decode"]
        match = re.search(r"-p([0-9]{2})-functional\.P$", prefill["rid"])
        assert match
        prefix_id = int(match[1])
        assert prefix_id not in identifiers and prefix_id in range(32)
        identifiers.add(prefix_id)
        assert prefill["rid"][:-1] == decode["rid"][:-1]
        assert row["P_DP"] == prefill["global_dp"] == prefix_id % 4
        assert row["D_DP"] == decode["global_dp"] == prefix_id % 16
        dp[prefix_id % 16] += 1
        for response, outputs in [(prefill, 1), (decode, 600)]:
            assert response["pass"] and response["status"] == 200
            assert response["terminal_monotonic_ns"] >= response["dispatch_monotonic_ns"]
            native = response["response"]
            assert native["usage"]["prompt_tokens"] == 512
            assert native["usage"]["completion_tokens"] == outputs
            assert native["choices"][0]["finish_reason"] == "length"
            assert native["choices"][0]["text"]
            assert hashlib.sha256(native["choices"][0]["text"].encode()).hexdigest() == response["text_sha256"]
            assert response["retry"] is False
            assert isinstance(response["vllm_short_reference_text_equal"], bool)
            assert "not" in response["vllm_short_reference_text_scope"].lower()
            assert response["vllm_short_reference_text_scope"].lower().find("whole") >= 0
            assert isinstance(response["vllm_short_reference_prefix16_text_equal"], bool)
            assert "diagnostic" in response["vllm_short_reference_prefix16_scope"].lower()
    assert identifiers == set(range(32))
    assert dp == collections.Counter({index: 2 for index in range(16)})
    sources = {
        "runs/" + contract.SHORT_RUN + "/receipt.json": sha("runs/" + contract.SHORT_RUN + "/receipt.json"),
        "runs/" + contract.SHORT_RUN + "/node-34.json": sha("runs/" + contract.SHORT_RUN + "/node-34.json"),
        contract.P_CAPACITY: contract.P_CAPACITY_SHA,
        contract.D_CAPACITY: contract.D_CAPACITY_SHA,
        contract.STARTUP_RESULT: sha(contract.STARTUP_RESULT),
        STARTUP_HELPER: STARTUP_HELPER_SHA,
    }
    capture_path = ROOT / "runs" / contract.FUNCTIONAL_CAPTURE / "receipt.json"
    assert capture_path.is_file(), "post-functional capture is pending; no runtime/data qualification"
    capture_receipt = json.loads(capture_path.read_text())
    assert capture_receipt["pass"] is True and capture_receipt["mode"] == "postfunctional"
    worker_rows = []
    graph_delta = {}
    productive = {}
    per_node = {}
    parent_environments = {}
    first_use_by_node = {}
    worker_env_by_node = {}
    worker_env_mask_by_node = {}
    non_sdk_by_node = {}
    for node in contract.D_NODES:
        after_rel = "runs/" + contract.FUNCTIONAL_CAPTURE + "/node-" + str(node) + ".json"
        before_rel = "runs/" + contract.STARTUP_CAPTURE + "/node-" + str(node) + ".json"
        after_path, after_frame, after = contract.load_native_capture(after_rel)
        before_path, before_frame, before = contract.load_native_capture(before_rel)
        sources[after_rel] = sha(after_rel)
        sources[before_rel] = sha(before_rel)
        expected = d_capacity["providers"][str(node)]
        assert after["container_id"] == before["container_id"] == expected["container_id"]
        before_events = contract.persistent_events(before)
        after_events = contract.persistent_events(after)
        before_init_rows = [row for row in before_events if row["event"] == "initialize_complete"]
        after_init_rows = [row for row in after_events if row["event"] == "initialize_complete"]
        assert len(after_init_rows) == len(before_init_rows) == 16
        init = {row["pid"] for row in after_init_rows}
        assert len(init) == 16
        assert {row["pid"] for row in before_init_rows} == init, "SDMA worker PID set changed"
        before_lifecycle = capture_lifecycle(before, node, before_init_rows)
        after_lifecycle = capture_lifecycle(after, node, after_init_rows)
        before_parent = actual_parent_environment(before, node)
        after_parent = actual_parent_environment(after, node)
        assert before_parent["actual"] == after_parent["actual"], "actual model parent environment changed"
        eager = after["D_eager_input_audit"]
        assert isinstance(eager, dict), "D eager audit missing"
        assert eager["applicable"] is True and eager["coverage_pass"] is True
        coverage = eager["coverage"]
        assert coverage["all_16_workers"] is True
        assert coverage["all_flag_true"] is True
        assert coverage["all_clone_ok"] is True
        assert coverage["all_bs2_draft_decode"] is True
        assert coverage["all_out_cache_loc_tensor_metadata"] is True
        assert coverage["no_row_errors"] is True
        workers = eager["worker_environment_and_identity"]
        assert isinstance(workers, list) and len(workers) == 16
        assert {row["namespace_pid"] for row in workers} == init
        assert len({row["host_pid"] for row in workers}) == 16
        assert sorted(row["device_id"] for row in workers) == list(range(16))
        assert all(row["alive"] and row["cgroup_contains_current_cid"] for row in workers)
        assert all(row["other_environment_retained"] is False for row in workers)
        assert all(row["full_environment_hashed"] is False for row in workers)
        first_use = STARTUP.first_use_coverage(eager, node, init)
        assert first_use["actual_sdk_workers"] == 16
        assert first_use["workers_with_native_no_copy_get"] == 16
        assert first_use["workers_with_clone_distinct"] == 16
        assert first_use["bs2_observed_workers"] == 16
        worker_env, worker_env_mask = STARTUP.worker_initial_environment(workers, node)
        assert len(worker_env) == 16
        records_by_pid = collections.defaultdict(list)
        for item in eager["records"]:
            row = item["row"]
            assert row["schema"] == "glm53.L17.eager_input_audit.v2"
            assert row["audit_version"] == "v2" and row["project_id"] == "glm53"
            assert row["source_sha256"] == row["eager_source_sha256"] == contract.EAGER_SOURCE_SHA
            assert row["environ_source_sha256"] == contract.ENVIRON_SOURCE_SHA
            assert row["node"] == node
            assert row["native_no_copy_get"] is True and row["clone_distinct"] is True
            assert type(row["dataclass_field_count"]) is int and row["dataclass_field_count"] > 0
            assert sum(row["field_verdicts"].values()) == row["dataclass_field_count"]
            assert set(row["allowlisted"]) == set(contract.AUDIT_ALLOWLIST_FIELDS)
            records_by_pid[row["pid"]].append(row)
        assert set(records_by_pid) == init
        by_pid = {}
        for row in eager["out_cache_loc_tensors"]:
            pid = row["pid"]
            assert pid in init and pid not in by_pid
            meta = row["out_cache_loc"]
            assert isinstance(meta["shape"], list) and isinstance(meta["stride"], list)
            assert len(meta["shape"]) == len(meta["stride"])
            assert all(type(value) is int and value >= 0 for value in meta["shape"])
            assert all(type(value) is int for value in meta["stride"])
            assert isinstance(meta["dtype"], str) and meta["dtype"]
            assert isinstance(meta["device"], str) and meta["device"]
            by_pid[pid] = meta
        assert len(by_pid) == 16 and set(by_pid) == init
        for worker in workers:
            pid = worker["namespace_pid"]
            required = [
                row for row in records_by_pid[pid]
                if row["is_draft_worker"] is True and row["forward_mode"] == "DECODE" and row["batch_size"] == 2
            ]
            assert required, "No actual bs2 draft-DECODE first-use row"
            assert any(row["native_no_copy_get"] is True and row["clone_distinct"] is True for row in required)
            assert pid in by_pid
            worker_rows.append({
                "node": node,
                "namespace_pid": pid,
                "host_pid": worker["host_pid"],
                "device_id": worker["device_id"],
                "is_draft_worker": True,
                "forward_mode": "DECODE",
                "batch_size": 2,
                "native_no_copy_get": True,
                "clone_distinct": True,
                "out_cache_loc": by_pid[pid],
                "required_rows": len(required),
            })
        before_counts = graph_counts(before)
        after_counts = graph_counts(after)
        expected_ranks = set(range(16)) if node == 36 else set(range(16, 32))
        assert set(before_counts) == set(after_counts) == expected_ranks
        assert all(after_counts[rank] > before_counts[rank] for rank in expected_ranks)
        graph_delta[str(node)] = {
            str(rank): after_counts[rank] - before_counts[rank] for rank in sorted(expected_ranks)
        }
        before_aligned = contract.alignment_events(before)
        after_aligned = contract.alignment_events(after)
        non_sdk = non_sdk_alignment_evidence(before_aligned, after_aligned, init)
        for pid in init:
            before_rows = [row for row in before_aligned if row["pid"] == pid]
            rows = [row for row in after_aligned if row["pid"] == pid]
            assert rows[:len(before_rows)] == before_rows, "SDK alignment rows are not append-only"
            assert all(row["revision"] == "glm53-C32-target-graph-fullmesh-v2" for row in rows)
            assert any(row["event"] == "draft_graph_disabled" for row in rows)
            for method in ["_npu_low_latency_dispatch", "_npu_low_latency_combine"]:
                found = [
                    row for row in rows
                    if row["event"] == "ops_first_return" and row["method"] == method
                ]
                assert len(found) == 1 and found[0]["group_size"] == 32
                assert found[0]["comm_alg"] == "fullmesh_v1"
            decisions = [row for row in rows if row["event"] == "target_graph_decision"]
            fresh = [
                row for row in decisions
                if row["eligible"] is True and row["mode"] == "TARGET_VERIFY" and row["raw_bs"] > 0 and not row["is_draft"]
            ]
            assert fresh, "Fresh productive target Graph eligibility required on every D SDK worker"
            assert not any(row["event"] == "target_graph_replay_return" for row in rows)
            rank = next(row["rank"] for row in rows if row["event"] == "ops_first_return")
            productive[f"{node}:{pid}"] = {
                "node": node,
                "pid": pid,
                "rank": rank,
                "productive_TARGET_VERIFY_decisions": len(fresh),
                "raw_bs_histogram": dict(collections.Counter(row["raw_bs"] for row in fresh)),
            }
        parent_environments[str(node)] = after_parent
        first_use_by_node[str(node)] = first_use
        worker_env_by_node[str(node)] = worker_env
        worker_env_mask_by_node[str(node)] = worker_env_mask
        non_sdk_by_node[str(node)] = non_sdk
        per_node[str(node)] = {
            "container_id": after["container_id"],
            "SDMA_lifecycle_before": before_lifecycle,
            "SDMA_lifecycle_after": after_lifecycle,
            "actual_parent_environment": after_parent,
            "first_use": first_use,
            "worker_initial_environment": worker_env,
            "worker_initial_environment_mask_counts": worker_env_mask,
            "worker_initial_environment_scope": "Raw /proc exec initial bytes only; missing keys remain missing and are not live EnvBool state. Live evidence is the per-worker first-use rows.",
            "non_SDK_alignment_installation_metadata": non_sdk,
        }
    assert len(worker_rows) == 32
    assert len({(row["node"], row["namespace_pid"]) for row in worker_rows}) == 32
    assert sorted(row["device_id"] for row in worker_rows) == sorted(list(range(16)) * 2)
    assert set(graph_delta) == {"36", "41"}
    assert len(productive) == 32
    assert {row["rank"] for row in productive.values()} == set(range(32))
    assert sum(item["rows"] for item in non_sdk_by_node.values()) == 13
    output = {
        "project_id": "glm53",
        "framework": "sglang",
        "pass": True,
        "status": "L17_SHORT_FUNCTIONAL_AND_D_EAGER_AUDIT_COMPLETE",
        "qualification": "C32functional/all16D-DP2each/countonly; D bs2 draft-DECODE coverage and productive Graph deltas verified. Worker /proc initial environment is raw/mask only; live evidence is first-use rows. Two80K semantic/native cache/SDMA gates still required.",
        "logical_pairs": 32,
        "backend_attempts": 64,
        "successful_backend_inferences": 64,
        "successful_output_tokens": 19232,
        "D_output_tokens_per_case": 600,
        "P_output_tokens_per_case": 1,
        "D_DP_logical_cases": dict(dp),
        "formal_requests": 0,
        "retries": 0,
        "runtime_CIDs": {str(node): (p_capacity if node in contract.P_NODES else d_capacity)["providers"][str(node)]["container_id"] for node in contract.ALL_NODES},
        "D_eager_input_acceptance": {
            "workers": len(worker_rows),
            "all_32_flag_true": all(row["native_no_copy_get"] for row in worker_rows),
            "all_32_clone_distinct": all(row["clone_distinct"] for row in worker_rows),
            "all_32_draft_DECODE_bs2": all(row["is_draft_worker"] and row["forward_mode"] == "DECODE" and row["batch_size"] == 2 for row in worker_rows),
            "all_32_out_cache_loc_metadata": all(row["out_cache_loc"] for row in worker_rows),
            "global_join_scope": "COMPOSITE_NODE_PLUS_NAMESPACE_PID; 32 unique (node, namespace_pid) pairs, never pooled by PID alone",
            "local_capture_status_scope": "Each node capture reports LOCAL16 only; global32 is derived solely by this composite join.",
            "worker_rows": worker_rows,
            "scope": "Actual per-PID first-use rows joined to the 16 SDK workers on each D node. out_cache_loc metadata is reported as recorded; no contiguity or filled field is claimed.",
        },
        "D_parent_environment": parent_environments,
        "D_first_use_coverage": first_use_by_node,
        "D_worker_initial_environment": worker_env_by_node,
        "D_worker_initial_environment_mask_counts": worker_env_mask_by_node,
        "D_worker_initial_environment_scope": "Raw /proc exec initial bytes with expected-match mask; missing keys stay missing and are not filled. This is not live EnvBool state and not first-use proof.",
        "D_non_SDK_alignment_installation_metadata": non_sdk_by_node,
        "D_non_SDK_alignment_scope": "Identical startup-v3/post-functional installation-only rows for non-SDK PIDs; only ops_compat_installed and target_graph_audit_installed; never productive worker proof.",
        "D_native_target_graph_delta_by_rank": graph_delta,
        "D_productive_graph_decisions": productive,
        "D_graph_scope": "Startup capture to post-functional capture, per global EP rank; all32 counters strictly increased.",
        "short_run_artifacts": result["artifacts"],
        "capture_receipt": capture_receipt,
        "source_evidence_sha256": sources,
        "external_model_requests": 64 + 0,
        "SDK_init_calls": 0,
        "new_profiles": 0,
        "formal_qualified": False,
        "runtime_data_qualification": "PENDING_ACTUAL_POST_FUNCTIONAL_CAPTURE_AND_LONG_COMPLETION",
    }
    target = ROOT / contract.SHORT_RESULT
    assert not target.exists(), "Preserve existing short functional result"
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "pass": True,
        "short_backend": 64,
        "short_output": 19232,
        "D_bs2_workers": len(worker_rows),
        "graph_ranks": sum(len(value) for value in graph_delta.values()),
        "non_SDK_alignment_rows": sum(item["rows"] for item in non_sdk_by_node.values()),
        "output": contract.SHORT_RESULT,
        "sha256": sha(contract.SHORT_RESULT),
    }))


if __name__ == "__main__":
    main()
