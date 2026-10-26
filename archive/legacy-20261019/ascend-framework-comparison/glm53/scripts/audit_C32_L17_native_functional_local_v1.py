"""DRAFT_PENDING_PRIMARY_REVIEW: independent L17 short functional/D eager audit.

Uses the actual short cohort and the post-functional native capture only.  It
requires exactly 32 logical / 64 backend requests, two per D DP, 19232 output
tokens, all32 actual draft-DECODE bs2 first-use rows with real out_cache_loc
metadata, and per-rank productive Graph deltas during the functional window.
It never fills a missing row, tensor field or Graph delta.
"""
import collections
import hashlib
import json
import re
from pathlib import Path

import glm53_C32_L17_pipeline_contract_v2 as contract

ROOT = Path(__file__).resolve().parents[1]


def sha(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


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


def main():
    p_capacity, d_capacity = contract.check_capacity_pair()
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
    }
    capture_receipt = json.loads((ROOT / "runs" / contract.FUNCTIONAL_CAPTURE / "receipt.json").read_text())
    assert capture_receipt["pass"] and capture_receipt["mode"] == "postfunctional"
    worker_rows = []
    graph_delta = {}
    productive = {}
    for node in contract.D_NODES:
        after_rel = "runs/" + contract.FUNCTIONAL_CAPTURE + "/node-" + str(node) + ".json"
        before_rel = "runs/" + contract.STARTUP_CAPTURE + "/node-" + str(node) + ".json"
        after_path, after_frame, after = contract.load_native_capture(after_rel)
        before_path, before_frame, before = contract.load_native_capture(before_rel)
        sources[after_rel] = sha(after_rel)
        sources[before_rel] = sha(before_rel)
        expected = d_capacity["providers"][str(node)]
        assert after["container_id"] == before["container_id"] == expected["container_id"]
        eager = after["D_eager_input_audit"]
        assert eager["applicable"] is True and eager["coverage_pass"] is True
        coverage = eager["coverage"]
        assert coverage["workers"] == 16
        assert coverage["all_16_workers"] and coverage["all_flag_true"]
        assert coverage["all_clone_ok"] and coverage["all_bs2_draft_decode"]
        assert coverage["all_out_cache_loc_tensor_metadata"] and coverage["no_row_errors"]
        assert not eager["file_errors"]
        assert len(eager["worker_environment_and_identity"]) == 16
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
        assert set(records_by_pid) == {row["namespace_pid"] for row in eager["worker_environment_and_identity"]}
        by_pid = {}
        for row in eager["out_cache_loc_tensors"]:
            pid = row["pid"]
            assert pid not in by_pid
            meta = row["out_cache_loc"]
            assert isinstance(meta["shape"], list) and isinstance(meta["stride"], list)
            assert len(meta["shape"]) == len(meta["stride"])
            assert all(type(value) is int and value >= 0 for value in meta["shape"])
            assert all(type(value) is int for value in meta["stride"])
            assert isinstance(meta["dtype"], str) and meta["dtype"]
            assert isinstance(meta["device"], str) and meta["device"]
            by_pid[pid] = meta
        assert len(by_pid) == 16
        for worker in eager["worker_environment_and_identity"]:
            pid = worker["namespace_pid"]
            env = worker["selected_environment"]
            assert env["SGLANG_EAGER_INPUT_NO_COPY"] == "1"
            assert env["GLM53_EAGER_INPUT_AUDIT"] == "1"
            assert env["GLM53_EAGER_INPUT_AUDIT_ROOT"] == contract.EAGER_ROOT_IN_CONTAINER
            assert env["GLM53_NODE"] == str(node)
            assert env["TASK_QUEUE_ENABLE"] == "0"
            assert env["HCCL_BUFFSIZE"] == "256"
            assert worker["alive"] and worker["cgroup_contains_current_cid"]
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
        init = {
            row["pid"] for row in contract.persistent_events(after)
            if row["event"] == "initialize_complete"
        }
        assert len(init) == 16
        aligned = contract.alignment_events(after)
        for pid in init:
            rows = [row for row in aligned if row["pid"] == pid]
            assert any(row["event"] == "draft_graph_disabled" for row in rows)
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
    assert len(worker_rows) == 32
    assert len({(row["node"], row["namespace_pid"]) for row in worker_rows}) == 32
    assert sorted(row["device_id"] for row in worker_rows) == sorted(list(range(16)) * 2)
    assert set(graph_delta) == {"36", "41"}
    assert len(productive) == 32
    assert {row["rank"] for row in productive.values()} == set(range(32))
    output = {
        "project_id": "glm53",
        "framework": "sglang",
        "pass": True,
        "status": "L17_SHORT_FUNCTIONAL_AND_D_EAGER_AUDIT_COMPLETE",
        "qualification": "C32functional/all16D-DP2each/countonly; D bs2 draft-DECODE coverage and productive Graph deltas verified. Two80K semantic/native cache/SDMA gates still required.",
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
        "output": contract.SHORT_RESULT,
        "sha256": sha(contract.SHORT_RESULT),
    }))


if __name__ == "__main__":
    main()
